# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from lmdeploy.messages import GenerationConfig
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

InputIdsType = List[int]
InputEmbsType = Union[None, List[Union[torch.Tensor, np.ndarray]]]
InputEmbRngsType = Union[None, List[Tuple[int, int]]]
PromptType = Union[str, List[Dict]]


class LogitsMixin:
    """Helper class to get logits, reward score and calculate ppl."""

    def get_reward_score(self, input_ids: List) -> List[float]:
        """
        Args:
            input_ids(List): a list of token_id or a list of token_id list or a tensor containing
                token_ids
        Return:
            reward score in a list. If the input_ids is a list of token_id, the return value
            is still a list with length 1.
        """
        supported_reward_models = ['InternLM2ForRewardModel', 'Qwen2ForRewardModel']
        if self.arch not in supported_reward_models:
            raise ValueError(f'{self.arch} is not in reward model list: {supported_reward_models}')
        assert isinstance(input_ids, List)
        assert all(isinstance(x, int) for x in input_ids) or all(isinstance(x, List) for x in input_ids)
        # Make input_ids a list of token_id list
        input_ids = [input_ids] if isinstance(input_ids[0], int) else input_ids
        logits = self._run(coro=self._async_get_logits(input_ids=input_ids)).result()
        logits = [x.squeeze() for x in logits]
        scores = [x[-1].cpu().item() for x in logits]
        return scores

    async def _async_get_reward_score(self, input_ids: List) -> List[float]:
        """Async version of get_reward_score."""
        supported_reward_models = ['InternLM2ForRewardModel', 'Qwen2ForRewardModel']
        if self.arch not in supported_reward_models:
            raise ValueError(f'{self.arch} is not in reward model list: {supported_reward_models}')
        assert isinstance(input_ids, List)
        assert all(isinstance(x, int) for x in input_ids) or all(isinstance(x, List) for x in input_ids)
        # Make input_ids a list of token_id list
        input_ids = [input_ids] if isinstance(input_ids[0], int) else input_ids

        logits = await self._async_get_logits(input_ids=input_ids)

        logits = [x.squeeze() for x in logits]
        scores = [x[-1].cpu().item() for x in logits]
        return scores

    async def _async_get_logits(self,
                                input_ids,
                                steps: List[int] = None,
                                sequence_start: bool = True,
                                sequence_end: bool = True) -> List[torch.Tensor]:
        assert input_ids and all(isinstance(_, List) for _ in input_ids)
        assert steps is None or (len(steps) == len(input_ids))

        logits = [None] * len(input_ids)

        async def _proc(i):
            async with self.model_inst(session_id=i) as inst:
                input_len = len(input_ids[i])
                # TODO(lvhan): Fix the ugly code later on
                max_new_tokens = 1 if self.backend == 'turbomind' else 0
                # The reason to set `top_k=1` is that pt engine crashes at top_k sampling stage
                # when perform inference on a reward model.
                gen_config = GenerationConfig(max_new_tokens=max_new_tokens, output_logits='all', top_k=1)
                async with self.safe_run(inst,
                                         session_id=i,
                                         input_ids=input_ids[i],
                                         gen_config=gen_config,
                                         stream_output=False,
                                         sequence_start=sequence_start,
                                         sequence_end=sequence_end,
                                         step=steps[i] if steps else 0) as gen:
                    async for outputs in gen:
                        pass
                    logits[i] = outputs.logits[:input_len, :]

        session_ids = list(range(len(input_ids)))
        tasks = [_proc(i) for i in range(len(input_ids))]
        await asyncio.gather(*tasks)
        if sequence_end and self.backend == 'pytorch':
            for session_id in session_ids:
                await self.end_session(session_id)
        return logits

    def get_ppl(self, input_ids: Union[List[int], List[List[int]]]) -> List[float]:
        """Get perplexity scores given a list of input tokens that have to be
        of the same length.

        Args:
            input_ids (Union[List[int], List[List[int]]]): the batch of
                input token ids

        Returns:
            List[float]: A list of perplexity scores.
        """
        assert isinstance(input_ids, List)
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]
        assert all(len(_) > 1 for _ in input_ids)

        # TODO: a better way to determine `max_input_len`, at most allocate
        # 2G mem for logits with shape [bs, max_input_len, vocab_size]
        vocab_size = self.hf_tm_cfg.vocab_size
        max_input_len = 2 * 1024**3 // (vocab_size * 4)
        sizes = [len(_) for _ in input_ids]
        result = []
        sorted_index_values = sorted(list(enumerate(sizes)), key=lambda x: x[1], reverse=True)
        sizes = [value for index, value in sorted_index_values]
        indices = [index for index, value in sorted_index_values]
        logger.info(f'sorted sizes: {sizes}')
        logger.info(f'sorted indices: {indices}')
        for (start, end) in self._batch_iterator(sizes, max_input_len):
            logger.info(f'start: {start}, end: {end}')
            if start == end:
                _input_ids = input_ids[indices[start]]
                res = self._get_long_text_ppl(input_ids=_input_ids, max_input_len=max_input_len)
                result.append(res)
            else:
                _input_ids = [input_ids[indices[i]] for i in range(start, end)]
                res = self._get_ppl(
                    input_ids=_input_ids,
                    max_input_len=max_input_len,
                )
                result.extend(res)
        output = list(range(len(result)))
        for index, sorted_index in enumerate(indices):
            output[sorted_index] = result[index]
        return output

    def _batch_iterator(self, sizes, max_value):
        """Return an iterator that calculates intervals (start, end) of a
        descend-order list, in which the sum of values in the range is the
        maximum number not less than max_value. By "the sum of values",

        here it means $$len(sizes[start:end]) * sizes[start]$$
        """
        i = 0
        while i < len(sizes):
            current_sum = 0
            start_index = i

            while i < len(sizes) and current_sum + sizes[start_index] <= max_value:
                current_sum += sizes[start_index]
                i += 1

            yield (start_index, i)
            if i > start_index:
                continue
            else:
                i += 1

    def _get_long_text_ppl(self, input_ids, max_input_len):
        assert all(isinstance(_, int) for _ in input_ids)
        seq_len = len(input_ids)
        assert seq_len > max_input_len
        logger.info(f'get long text ppl: seq_len {seq_len}')

        losses = []
        target_counts = []
        for i in range(0, seq_len, max_input_len):
            token_ids = input_ids[i:i + max_input_len]
            step = [i]
            # shift token_ids by 1 to the left
            target_ids = input_ids[i + 1:i + 1 + max_input_len]
            loss = self._get_ppl(input_ids=[token_ids],
                                 max_input_len=len(token_ids),
                                 target_ids=[target_ids],
                                 steps=step,
                                 sequence_start=(i == 0),
                                 sequence_end=False)
            losses.extend(loss)
            target_counts.append(len(target_ids))
        losses = [loss * target_count for loss, target_count in zip(losses, target_counts)]
        loss_sum = sum(losses)
        target_count = sum(target_counts)
        return loss_sum / target_count

    def _get_ppl(self,
                 input_ids,
                 max_input_len,
                 target_ids=None,
                 steps=None,
                 sequence_start: bool = True,
                 sequence_end: bool = True):
        assert (isinstance(input_ids, List) and all(isinstance(_, List) for _ in input_ids))
        assert steps is None or len(steps) == len(input_ids)
        assert target_ids is None or len(target_ids) == len(input_ids)

        lens = [len(_) for _ in input_ids]
        total_len = sum(lens)
        assert sum(lens) <= max_input_len

        logger.info(f'get_ppl: bs: {len(input_ids)}, lens: {lens}, '
                    f'total_len: {total_len}, steps: {steps}')
        torch.cuda.empty_cache()

        logits = self._run(coro=self._async_get_logits(
            input_ids=input_ids, steps=steps, sequence_start=sequence_start, sequence_end=sequence_end)).result()
        padding_token_id = -100
        if target_ids is None:
            target_ids = [x[1:] + [padding_token_id] for x in input_ids]
        else:
            target_ids = [
                target_ids[i] + [padding_token_id] if len(target_ids[i]) < len(input_ids[i]) else target_ids[i]
                for i in range(len(input_ids))
            ]
        target_ids = [torch.Tensor(torch.LongTensor(_target_ids)) for _target_ids in target_ids]

        result = []
        for _logits, _target_ids in zip(logits, target_ids):
            _logits = _logits.float()
            vocab_size = _logits.shape[-1]
            _target_ids = _target_ids.to(_logits.device)
            target_mask = _target_ids != padding_token_id
            # compute cross entropy loss
            flat_logits = _logits.contiguous().view(-1, vocab_size)
            flat_target_ids = _target_ids.contiguous().view(-1)
            flat_loss_matrix = torch.nn.functional.cross_entropy(flat_logits,
                                                                 flat_target_ids,
                                                                 reduction='none',
                                                                 ignore_index=padding_token_id)
            loss = flat_loss_matrix.sum()
            target_count = target_mask.sum()
            result.append(loss.item() / target_count.item())
        logger.info(f'ppl result: {result}')
        return result
    
    def get_dllm_ppl(self, input_ids: Union[List[int], List[List[int]]]) -> List[Dict]:
        """Get perplexity scores for DLLM models using greedy unmasking strategy.
        
        Args:
            input_ids (Union[List[int], List[List[int]]]): the batch of
                input token ids (ground truth sequences)
        
        Returns:
            List[Dict]: A list of dictionaries containing:
                - 'perplexity': float, the perplexity score
                - 'decode_orders': List[List[int]], decode order for each block
        """
        assert isinstance(input_ids, List)
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]
        assert all(len(_) > 1 for _ in input_ids)
        
        # Get DLLM configuration
        dllm_config = self.engine.model_config.dllm_block_length
        if dllm_config is None:
            raise ValueError("Model does not have DLLM configuration")
        
        block_size = dllm_config
        
        # Parallel processing across sequences
        result = self._run(coro=self._async_get_dllm_ppl_batch(input_ids, block_size)).result()
        
        return result


    async def _async_get_dllm_ppl_batch(self, input_ids_list: List[List[int]], block_size: int) -> List[Dict]:
        """Compute perplexity for multiple sequences in parallel.
        
        Args:
            input_ids_list (List[List[int]]): list of ground truth token id sequences
            block_size (int): DLLM block size
        
        Returns:
            List[Dict]: results for each sequence
        """
        import asyncio
        
        async def _process_single(seq_input_ids, seq_idx):
            ppl, decode_orders = await self._async_get_single_sequence_dllm_ppl(
                seq_input_ids, block_size, session_id_base=seq_idx * 10000
            )
            return {
                'perplexity': ppl,
                'decode_orders': decode_orders
            }
        
        tasks = [_process_single(seq_input_ids, idx) for idx, seq_input_ids in enumerate(input_ids_list)]
        results = await asyncio.gather(*tasks)
        
        return results


    async def _async_get_single_sequence_dllm_ppl(self, input_ids: List[int], block_size: int, 
                                                    session_id_base: int = 0) -> tuple:
        """Async version: Compute perplexity for a single sequence using DLLM greedy unmasking.
        
        Args:
            input_ids (List[int]): ground truth token ids
            block_size (int): DLLM block size
            session_id_base (int): base session id to avoid conflicts
        
        Returns:
            tuple: (perplexity, decode_orders)
        """
        import torch
        import numpy as np
        
        seq_len = len(input_ids)
        
        # Get mask token id
        mask_token_id = self.engine.model_config.dllm_mask_token
        
        # Pad sequence to multiple of block_size
        padding_len = (block_size - seq_len % block_size) % block_size
        if padding_len > 0:
            input_ids = input_ids + [mask_token_id] * padding_len
        
        num_blocks = len(input_ids) // block_size
        
        all_log_probs = []
        all_decode_orders = []
        
        # Process each block
        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_end = block_start + block_size
            ground_truth_block = input_ids[block_start:block_end]
            
            # Determine which positions in this block are valid (not padding)
            valid_positions = set()
            for pos in range(block_size):
                if block_start + pos < seq_len:
                    valid_positions.add(pos)
            
            # Skip if entire block is padding
            if len(valid_positions) == 0:
                break
            
            # Initialize block with all masks
            current_block = [mask_token_id] * block_size
            unmasked_positions = set()
            decode_order = []
            
            # Greedy unmasking: unmask one token at a time
            for unmask_step in range(len(valid_positions)):
                # Construct full input sequence
                full_input = input_ids[:block_start] + current_block
                
                # Get logits from model (use unique session_id)
                session_id = session_id_base + block_idx * 100 + unmask_step
                logits = await self._async_get_dllm_logits_single(
                    input_ids=full_input,
                    block_start=block_start,
                    block_size=block_size,
                    session_id=session_id
                )  # [block_size, vocab_size]
                
                # Find position with highest peak probability among masked AND valid positions
                max_peak_prob = -float('inf')
                next_unmask_pos = None
                
                for pos in range(block_size):
                    if pos in unmasked_positions:
                        continue
                    if pos not in valid_positions:
                        continue
                    
                    probs = torch.softmax(logits[pos], dim=-1)
                    peak_prob = probs.max().item()
                    
                    if peak_prob > max_peak_prob:
                        max_peak_prob = peak_prob
                        next_unmask_pos = pos
                
                if next_unmask_pos is None:
                    break
                
                decode_order.append(next_unmask_pos)
                
                # Get probability of ground truth token
                gt_token_id = ground_truth_block[next_unmask_pos]
                probs = torch.softmax(logits[next_unmask_pos], dim=-1)
                gt_prob = probs[gt_token_id].item()
                log_prob = np.log(gt_prob + 1e-10)
                all_log_probs.append(log_prob)
                
                # Unmask with ground truth
                current_block[next_unmask_pos] = gt_token_id
                unmasked_positions.add(next_unmask_pos)
            
            all_decode_orders.append(decode_order)
        
        # Compute perplexity
        avg_log_prob = np.mean(all_log_probs)
        perplexity = np.exp(-avg_log_prob)
        
        return perplexity, all_decode_orders


    async def _async_get_dllm_logits_single(self, 
                                         input_ids: List[int],
                                         block_start: int,
                                         block_size: int,
                                         session_id: int) -> torch.Tensor:
        """Get logits for a single forward pass.
        
        Args:
            input_ids (List[int]): input sequence with masks
            block_start (int): start position of the block
            block_size (int): size of the block
            session_id (int): unique session id
        
        Returns:
            torch.Tensor: logits for the block, shape [block_size, vocab_size]
        """
        from lmdeploy.messages import GenerationConfig
        
        async with self.model_inst(session_id=session_id) as inst:
            gen_config = GenerationConfig(
                max_new_tokens=0,
                output_logits='all',
                top_k=1
            )
            
            async with self.safe_run(inst,
                                    session_id=session_id,
                                    input_ids=input_ids,
                                    gen_config=gen_config,
                                    stream_output=False,
                                    sequence_start=True,
                                    sequence_end=True,
                                    step=0) as gen:
                async for outputs in gen:
                    pass
                
                # Extract logits for the target block
                block_end = block_start + block_size
                block_logits = outputs.logits[block_start:block_end, :]
                
                return block_logits