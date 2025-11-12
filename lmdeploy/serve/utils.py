# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from lmdeploy.messages import GenerationConfig
from lmdeploy.utils import get_logger

logger = get_logger("lmdeploy")

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
        supported_reward_models = ["InternLM2ForRewardModel", "Qwen2ForRewardModel"]
        if self.arch not in supported_reward_models:
            raise ValueError(
                f"{self.arch} is not in reward model list: {supported_reward_models}"
            )
        assert isinstance(input_ids, List)
        assert all(isinstance(x, int) for x in input_ids) or all(
            isinstance(x, List) for x in input_ids
        )
        # Make input_ids a list of token_id list
        input_ids = [input_ids] if isinstance(input_ids[0], int) else input_ids
        logits = self._run(coro=self._async_get_logits(input_ids=input_ids)).result()
        logits = [x.squeeze() for x in logits]
        scores = [x[-1].cpu().item() for x in logits]
        return scores

    async def _async_get_reward_score(self, input_ids: List) -> List[float]:
        """Async version of get_reward_score."""
        supported_reward_models = ["InternLM2ForRewardModel", "Qwen2ForRewardModel"]
        if self.arch not in supported_reward_models:
            raise ValueError(
                f"{self.arch} is not in reward model list: {supported_reward_models}"
            )
        assert isinstance(input_ids, List)
        assert all(isinstance(x, int) for x in input_ids) or all(
            isinstance(x, List) for x in input_ids
        )
        # Make input_ids a list of token_id list
        input_ids = [input_ids] if isinstance(input_ids[0], int) else input_ids

        logits = await self._async_get_logits(input_ids=input_ids)

        logits = [x.squeeze() for x in logits]
        scores = [x[-1].cpu().item() for x in logits]
        return scores

    async def _async_get_logits(
        self,
        input_ids,
        steps: List[int] = None,
        sequence_start: bool = True,
        sequence_end: bool = True,
    ) -> List[torch.Tensor]:
        assert input_ids and all(isinstance(_, List) for _ in input_ids)
        assert steps is None or (len(steps) == len(input_ids))

        logits = [None] * len(input_ids)

        async def _proc(i):
            async with self.model_inst(session_id=i) as inst:
                input_len = len(input_ids[i])
                # TODO(lvhan): Fix the ugly code later on
                max_new_tokens = 1 if self.backend == "turbomind" else 0
                # The reason to set `top_k=1` is that pt engine crashes at top_k sampling stage
                # when perform inference on a reward model.
                gen_config = GenerationConfig(
                    max_new_tokens=max_new_tokens, output_logits="all", top_k=1
                )
                async with self.safe_run(
                    inst,
                    session_id=i,
                    input_ids=input_ids[i],
                    gen_config=gen_config,
                    stream_output=False,
                    sequence_start=sequence_start,
                    sequence_end=sequence_end,
                    step=steps[i] if steps else 0,
                ) as gen:
                    async for outputs in gen:
                        pass
                    logits[i] = outputs.logits[:input_len, :]

        session_ids = list(range(len(input_ids)))
        tasks = [_proc(i) for i in range(len(input_ids))]
        await asyncio.gather(*tasks)
        if sequence_end and self.backend == "pytorch":
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
        sorted_index_values = sorted(
            list(enumerate(sizes)), key=lambda x: x[1], reverse=True
        )
        sizes = [value for index, value in sorted_index_values]
        indices = [index for index, value in sorted_index_values]
        logger.info(f"sorted sizes: {sizes}")
        logger.info(f"sorted indices: {indices}")
        for start, end in self._batch_iterator(sizes, max_input_len):
            logger.info(f"start: {start}, end: {end}")
            if start == end:
                _input_ids = input_ids[indices[start]]
                res = self._get_long_text_ppl(
                    input_ids=_input_ids, max_input_len=max_input_len
                )
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
        logger.info(f"get long text ppl: seq_len {seq_len}")

        losses = []
        target_counts = []
        for i in range(0, seq_len, max_input_len):
            token_ids = input_ids[i : i + max_input_len]
            step = [i]
            # shift token_ids by 1 to the left
            target_ids = input_ids[i + 1 : i + 1 + max_input_len]
            loss = self._get_ppl(
                input_ids=[token_ids],
                max_input_len=len(token_ids),
                target_ids=[target_ids],
                steps=step,
                sequence_start=(i == 0),
                sequence_end=False,
            )
            losses.extend(loss)
            target_counts.append(len(target_ids))
        losses = [
            loss * target_count for loss, target_count in zip(losses, target_counts)
        ]
        loss_sum = sum(losses)
        target_count = sum(target_counts)
        return loss_sum / target_count

    def _get_ppl(
        self,
        input_ids,
        max_input_len,
        target_ids=None,
        steps=None,
        sequence_start: bool = True,
        sequence_end: bool = True,
    ):
        assert isinstance(input_ids, List) and all(
            isinstance(_, List) for _ in input_ids
        )
        assert steps is None or len(steps) == len(input_ids)
        assert target_ids is None or len(target_ids) == len(input_ids)

        lens = [len(_) for _ in input_ids]
        total_len = sum(lens)
        assert sum(lens) <= max_input_len

        logger.info(
            f"get_ppl: bs: {len(input_ids)}, lens: {lens}, "
            f"total_len: {total_len}, steps: {steps}"
        )
        torch.cuda.empty_cache()

        logits = self._run(
            coro=self._async_get_logits(
                input_ids=input_ids,
                steps=steps,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
            )
        ).result()
        padding_token_id = -100
        if target_ids is None:
            target_ids = [x[1:] + [padding_token_id] for x in input_ids]
        else:
            target_ids = [
                target_ids[i] + [padding_token_id]
                if len(target_ids[i]) < len(input_ids[i])
                else target_ids[i]
                for i in range(len(input_ids))
            ]
        target_ids = [
            torch.Tensor(torch.LongTensor(_target_ids)) for _target_ids in target_ids
        ]

        result = []
        for _logits, _target_ids in zip(logits, target_ids):
            _logits = _logits.float()
            vocab_size = _logits.shape[-1]
            _target_ids = _target_ids.to(_logits.device)
            target_mask = _target_ids != padding_token_id
            # compute cross entropy loss
            flat_logits = _logits.contiguous().view(-1, vocab_size)
            flat_target_ids = _target_ids.contiguous().view(-1)
            flat_loss_matrix = torch.nn.functional.cross_entropy(
                flat_logits,
                flat_target_ids,
                reduction="none",
                ignore_index=padding_token_id,
            )
            loss = flat_loss_matrix.sum()
            target_count = target_mask.sum()
            result.append(loss.item() / target_count.item())
        logger.info(f"ppl result: {result}")
        return result

    def get_dllm_ppl(
        self, input_ids: Union[List[int], List[List[int]]], max_concurrent: int = None
    ) -> List[Dict]:
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

        # Default to a conservative concurrency limit
        if max_concurrent is None:
            max_concurrent = 1

        # Parallel processing across sequences
        result = self._run(
            coro=self._async_get_dllm_ppl_batch(input_ids, block_size, max_concurrent)
        ).result()

        return result

    async def _async_get_dllm_ppl_batch(
        self, input_ids_list: List[List[int]], block_size: int, max_concurrent: int
    ) -> List[Dict]:
        """Compute perplexity for multiple sequences in parallel.

        Args:
            input_ids_list (List[List[int]]): list of ground truth token id sequences
            block_size (int): DLLM block size

        Returns:
            List[Dict]: results for each sequence
        """
        import asyncio

        # Semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _process_single_with_semaphore(seq_input_ids, seq_idx):
            async with semaphore:
                (
                    seq_len,
                    nll,
                    decode_orders,
                ) = await self._async_get_single_sequence_dllm_ppl(
                    seq_input_ids, block_size, session_id_base=seq_idx * 10000
                )
                return {
                    "num_tokens": seq_len,
                    "nll": nll,
                    "perplexity": np.exp(nll),
                    "decode_orders": decode_orders,
                    "sequence_index": seq_idx,
                }

        tasks = [
            _process_single_with_semaphore(seq_input_ids, idx)
            for idx, seq_input_ids in enumerate(input_ids_list)
        ]
        results = await asyncio.gather(*tasks)

        # Sort by sequence_index to maintain input order
        results.sort(key=lambda x: x["sequence_index"])

        # Remove sequence_index from output
        for result in results:
            result.pop("sequence_index")

        return results

    async def _async_get_single_sequence_dllm_ppl(
        self, input_ids: List[int], block_size: int, session_id_base: int = 0
    ) -> tuple:
        """Compute perplexity for a single sequence using DLLM greedy unmasking.

        This version avoids conflicts with the engine's DLLM sequence management.
        """
        import torch
        import numpy as np
        from lmdeploy.utils import get_logger

        logger = get_logger("lmdeploy")

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

        # Use unique session ID for this sequence
        session_id = session_id_base

        # Process each block INDEPENDENTLY to avoid engine state conflicts
        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_end = block_start + block_size
            ground_truth_block = input_ids[block_start:block_end]

            # Determine valid positions
            valid_positions = set()
            for pos in range(block_size):
                if block_start + pos < seq_len:
                    valid_positions.add(pos)

            if len(valid_positions) == 0:
                break

            logger.debug(
                f"Processing block {block_idx}, valid_positions: {valid_positions}"
            )

            # Initialize block with all masks
            current_block = [mask_token_id] * block_size
            unmasked_positions = set()
            decode_order = []

            # Process this block with iterative unmasking
            for unmask_step in range(len(valid_positions)):
                # Construct input: previous blocks (ground truth) + current block state
                if block_idx == 0:
                    # First block: only current block
                    full_input = current_block
                else:
                    # Subsequent blocks: all previous ground truth + current block state
                    full_input = input_ids[:block_start] + current_block

                logger.debug(
                    f"Block {block_idx}, step {unmask_step}, input len: {len(full_input)}"
                )

                try:
                    # Get logits - use different session for each block+step to avoid state conflicts
                    step_session_id = session_id_base + block_idx * 1000 + unmask_step

                    logits = await self._async_get_dllm_logits_with_cache(
                        input_ids=full_input,
                        session_id=step_session_id,
                        sequence_start=True,  # Always start fresh to avoid state issues
                        sequence_end=True,  # Always end to clean up
                        step=0,  # Don't use KV cache for now
                    )

                    # Extract logits for current block (last block_size positions)
                    block_logits = logits[-block_size:, :]

                    logger.debug(
                        f"Got logits shape: {logits.shape}, block_logits shape: {block_logits.shape}"
                    )

                except Exception as e:
                    logger.error(f"Error in block {block_idx}, step {unmask_step}: {e}")
                    raise

                # Find position with highest peak probability among masked AND valid positions
                max_peak_prob = -float("inf")
                next_unmask_pos = None

                for pos in range(block_size):
                    if pos in unmasked_positions:
                        continue
                    if pos not in valid_positions:
                        continue

                    probs = torch.softmax(block_logits[pos], dim=-1)
                    peak_prob = probs.max().item()

                    if peak_prob > max_peak_prob:
                        max_peak_prob = peak_prob
                        next_unmask_pos = pos

                if next_unmask_pos is None:
                    logger.warning(
                        f"No next position found in block {block_idx}, step {unmask_step}"
                    )
                    break

                decode_order.append(next_unmask_pos)

                # Get probability of ground truth token
                gt_token_id = ground_truth_block[next_unmask_pos]
                probs = torch.softmax(block_logits[next_unmask_pos], dim=-1)
                gt_prob = probs[gt_token_id].item()
                log_prob = np.log(gt_prob + 1e-10)
                all_log_probs.append(log_prob)

                logger.debug(
                    f"Unmasked pos {next_unmask_pos}, gt_token: {gt_token_id}, "
                    f"gt_prob: {gt_prob:.4f}, peak_prob: {max_peak_prob:.4f}"
                )

                # Unmask with ground truth
                current_block[next_unmask_pos] = gt_token_id
                unmasked_positions.add(next_unmask_pos)

            all_decode_orders.append(decode_order)
            logger.info(f"Block {block_idx} decode order: {decode_order}")

        # Compute nll
        if len(all_log_probs) == 0:
            logger.error("No log probs collected!")
            return float("inf"), all_decode_orders

        nll = -np.mean(all_log_probs)

        logger.info(f"Sequence nll: {nll:.4f}, decode_orders: {all_decode_orders}")

        return seq_len, nll, all_decode_orders

    async def _async_get_dllm_logits_with_cache(
        self,
        input_ids: List[int],
        session_id: int,
        sequence_start: bool,
        sequence_end: bool,
        step: int,
    ) -> torch.Tensor:
        """Get logits with KV cache management for DLLM perplexity computation.

        Args:
            input_ids (List[int]): full input sequence
            session_id (int): session ID for KV cache tracking
            sequence_start (bool): whether this is the start of the sequence
            sequence_end (bool): whether to end the session after this forward pass
            step (int): current position in the KV cache

        Returns:
            torch.Tensor: logits for all input positions, shape [seq_len, vocab_size]
        """
        from lmdeploy.messages import GenerationConfig

        try:
            async with self.model_inst(session_id=session_id) as inst:
                gen_config = GenerationConfig(
                    max_new_tokens=0,  # We only want logits, no generation
                    output_logits="all",  # Get logits for all positions
                    top_k=1,
                    # Important: these flags prevent DLLM from managing state
                )

                async with self.safe_run(
                    inst,
                    session_id=session_id,
                    input_ids=input_ids,
                    gen_config=gen_config,
                    stream_output=False,
                    sequence_start=sequence_start,
                    sequence_end=sequence_end,
                    step=step,
                ) as gen:
                    outputs = None
                    async for out in gen:
                        outputs = out

                    # Check if we got valid outputs
                    if outputs is None:
                        raise ValueError(
                            f"Engine returned None for session {session_id}"
                        )

                    if outputs.logits is None:
                        raise ValueError(f"Logits is None for session {session_id}")

                    return outputs.logits

        except Exception as e:
            logger = get_logger("lmdeploy")
            logger.error(f"Error getting logits: {e}")
            raise


    # def get_dllm_ppl(
    #     self, input_ids: Union[List[int], List[List[int]]], batch_size: int = 4
    # ) -> List[Dict]:
    #     """Get perplexity using batched direct model access.

    #     Args:
    #         input_ids: Single sequence or list of sequences
    #         batch_size: Number of sequences to process together

    #     Returns:
    #         List of dictionaries with perplexity and decode_orders
    #     """
    #     assert isinstance(input_ids, List)
    #     if isinstance(input_ids[0], int):
    #         input_ids = [input_ids]

    #     dllm_block_length = self.engine.model_config.dllm_block_length
    #     if dllm_block_length is None:
    #         raise ValueError("Model does not have DLLM configuration")

    #     # Get model
    #     model = self._get_model()

    #     # Process in batches
    #     all_results = []
    #     for i in range(0, len(input_ids), batch_size):
    #         batch = input_ids[i : i + batch_size]
    #         batch_results = self._compute_dllm_ppl_direct(
    #             batch, dllm_block_length, model
    #         )

    #         for nll, decode_orders in batch_results:
    #             all_results.append({"nll": nll, "decode_orders": decode_orders})

    #     return all_results

    # def _get_model(self):
    #     """Get the underlying model."""
    #     if self.backend == 'pytorch':
    #         # Try different possible paths to access the model
    #         if hasattr(self.engine, 'model_agent'):
    #             return self.engine.model_agent.model
    #         elif hasattr(self.engine, 'model'):
    #             return self.engine.model
    #         elif hasattr(self.engine, 'engine_instance'):
    #             return self.engine.engine_instance.model
    #         else:
    #             # Inspect the engine to find the model
    #             logger = get_logger('lmdeploy')
    #             logger.error(f"Engine attributes: {dir(self.engine)}")
    #             raise AttributeError(
    #                 "Could not find model in engine. Please check engine structure. "
    #                 f"Available attributes: {[attr for attr in dir(self.engine) if not attr.startswith('_')]}"
    #             )
    #     else:
    #         raise NotImplementedError("Direct model access only supported for pytorch backend")

    # def _compute_dllm_ppl_direct(
    #     self, input_ids_batch: List[List[int]], block_size: int, model
    # ) -> List[tuple]:
    #     """Compute perplexity for a batch of sequences using direct model access.

    #     Args:
    #         input_ids_batch: List of sequences (each is List[int])
    #         block_size: DLLM block size
    #         model: The model instance
    #         tokenizer: The tokenizer

    #     Returns:
    #         List of (perplexity, decode_orders) tuples for each sequence
    #     """
    #     import torch
    #     import numpy as np

    #     batch_size = len(input_ids_batch)
    #     mask_token_id = model.model_config.dllm_mask_token
    #     device = next(model.parameters()).device

    #     # Get sequence lengths and pad to block_size
    #     seq_lengths = [len(seq) for seq in input_ids_batch]
    #     padded_sequences = []

    #     for input_ids in input_ids_batch:
    #         seq_len = len(input_ids)
    #         padding_len = (block_size - seq_len % block_size) % block_size
    #         if padding_len > 0:
    #             input_ids = input_ids + [mask_token_id] * padding_len
    #         padded_sequences.append(input_ids)

    #     # Calculate number of blocks for each sequence
    #     num_blocks_per_seq = [len(seq) // block_size for seq in padded_sequences]
    #     max_num_blocks = max(num_blocks_per_seq)

    #     # Initialize tracking for each sequence
    #     all_log_probs = [[] for _ in range(batch_size)]
    #     all_decode_orders = [[] for _ in range(batch_size)]
    #     sequence_active = [True] * batch_size  # Track which sequences are still active
    #     current_block_idx = [0] * batch_size  # Current block index for each sequence

    #     # Past KV cache for each sequence (previous completed blocks)
    #     past_key_values_batch = [None] * batch_size

    #     with torch.inference_mode():
    #         # Loop 1: Iterate over maximum number of blocks
    #         for global_block_idx in range(max_num_blocks):
    #             # Determine which sequences are processing this block
    #             active_in_this_block = []
    #             for seq_idx in range(batch_size):
    #                 if (
    #                     sequence_active[seq_idx]
    #                     and current_block_idx[seq_idx] < num_blocks_per_seq[seq_idx]
    #                 ):
    #                     active_in_this_block.append(seq_idx)

    #             if len(active_in_this_block) == 0:
    #                 break  # All sequences done

    #             # Initialize current blocks for active sequences
    #             current_blocks = {}  # seq_idx -> current_block
    #             unmasked_positions = {}  # seq_idx -> set of unmasked positions
    #             decode_orders = {}  # seq_idx -> decode order for this block
    #             valid_positions_map = {}  # seq_idx -> set of valid positions

    #             for seq_idx in active_in_this_block:
    #                 block_idx = current_block_idx[seq_idx]
    #                 block_start = block_idx * block_size

    #                 # Initialize block with all masks
    #                 current_blocks[seq_idx] = [mask_token_id] * block_size
    #                 unmasked_positions[seq_idx] = set()
    #                 decode_orders[seq_idx] = []

    #                 # Determine valid positions (non-padding)
    #                 valid_positions = set()
    #                 seq_len = seq_lengths[seq_idx]
    #                 for pos in range(block_size):
    #                     if block_start + pos < seq_len:
    #                         valid_positions.add(pos)
    #                 valid_positions_map[seq_idx] = valid_positions

    #                 # If no valid positions, mark sequence as inactive
    #                 if len(valid_positions) == 0:
    #                     sequence_active[seq_idx] = False

    #             # Remove sequences with no valid positions
    #             active_in_this_block = [
    #                 s for s in active_in_this_block if sequence_active[s]
    #             ]

    #             if len(active_in_this_block) == 0:
    #                 continue

    #             # Loop 2: Unmask tokens one by one (up to block_size iterations)
    #             for unmask_step in range(block_size):
    #                 # Check if any sequence still has positions to unmask
    #                 sequences_to_process = []
    #                 for seq_idx in active_in_this_block:
    #                     if len(unmasked_positions[seq_idx]) < len(
    #                         valid_positions_map[seq_idx]
    #                     ):
    #                         sequences_to_process.append(seq_idx)

    #                 if len(sequences_to_process) == 0:
    #                     break  # All active sequences finished unmasking this block

    #                 # Prepare batch input
    #                 batch_input_ids = []
    #                 batch_past_kvs = []
    #                 seq_idx_mapping = []  # Maps batch position to original seq_idx

    #                 for seq_idx in sequences_to_process:
    #                     block_idx = current_block_idx[seq_idx]
    #                     block_start = block_idx * block_size

    #                     # Construct input: [previous_blocks + current_block]
    #                     prev_blocks = padded_sequences[seq_idx][:block_start]
    #                     full_input = prev_blocks + current_blocks[seq_idx]

    #                     batch_input_ids.append(full_input)
    #                     batch_past_kvs.append(past_key_values_batch[seq_idx])
    #                     seq_idx_mapping.append(seq_idx)

    #                 # Handle different past_kv lengths by padding or separate forward passes
    #                 # For simplicity, process sequences with same history length together
    #                 grouped_by_history = {}
    #                 for i, seq_idx in enumerate(seq_idx_mapping):
    #                     history_len = len(
    #                         padded_sequences[seq_idx][
    #                             : current_block_idx[seq_idx] * block_size
    #                         ]
    #                     )
    #                     if history_len not in grouped_by_history:
    #                         grouped_by_history[history_len] = []
    #                     grouped_by_history[history_len].append((i, seq_idx))

    #                 # Process each group
    #                 all_logits = {}  # seq_idx -> logits

    #                 for history_len, group in grouped_by_history.items():
    #                     group_indices = [i for i, _ in group]
    #                     group_seq_indices = [seq_idx for _, seq_idx in group]

    #                     # Prepare tensors for this group
    #                     group_input_ids = [batch_input_ids[i] for i in group_indices]

    #                     # Pad to same length within group
    #                     max_len = max(len(ids) for ids in group_input_ids)
    #                     padded_group_input = []
    #                     for ids in group_input_ids:
    #                         padded = ids + [mask_token_id] * (max_len - len(ids))
    #                         padded_group_input.append(padded)

    #                     input_tensor = torch.tensor(
    #                         padded_group_input, dtype=torch.long, device=device
    #                     )

    #                     # For now, process without past_kv to simplify
    #                     # (Full implementation would need to handle past_kv batching properly)
    #                     outputs = model(
    #                         input_ids=input_tensor,
    #                         past_key_values=None,  # Simplified: recompute each time
    #                         use_cache=False,
    #                         return_dict=True,
    #                     )

    #                     # Extract logits for each sequence in group
    #                     for local_idx, seq_idx in enumerate(group_seq_indices):
    #                         block_idx = current_block_idx[seq_idx]
    #                         block_start = block_idx * block_size

    #                         # Get logits for current block
    #                         seq_logits = outputs.logits[local_idx]
    #                         block_logits = seq_logits[
    #                             -block_size:, :
    #                         ]  # Last block_size positions
    #                         all_logits[seq_idx] = block_logits

    #                 # Now process logits for each sequence
    #                 for seq_idx in sequences_to_process:
    #                     if seq_idx not in all_logits:
    #                         continue

    #                     block_logits = all_logits[seq_idx]
    #                     block_idx = current_block_idx[seq_idx]
    #                     block_start = block_idx * block_size
    #                     ground_truth_block = padded_sequences[seq_idx][
    #                         block_start : block_start + block_size
    #                     ]

    #                     # Find position with highest peak probability
    #                     max_peak_prob = -float("inf")
    #                     next_unmask_pos = None

    #                     for pos in range(block_size):
    #                         if pos in unmasked_positions[seq_idx]:
    #                             continue
    #                         if pos not in valid_positions_map[seq_idx]:
    #                             continue

    #                         probs = torch.softmax(block_logits[pos], dim=-1)
    #                         peak_prob = probs.max().item()

    #                         if peak_prob > max_peak_prob:
    #                             max_peak_prob = peak_prob
    #                             next_unmask_pos = pos

    #                     if next_unmask_pos is None:
    #                         continue

    #                     # Record decode order
    #                     decode_orders[seq_idx].append(next_unmask_pos)

    #                     # Collect ground truth probability
    #                     gt_token_id = ground_truth_block[next_unmask_pos]
    #                     probs = torch.softmax(block_logits[next_unmask_pos], dim=-1)
    #                     gt_prob = probs[gt_token_id].item()
    #                     log_prob = np.log(gt_prob + 1e-10)
    #                     all_log_probs[seq_idx].append(log_prob)

    #                     # Unmask with ground truth
    #                     current_blocks[seq_idx][next_unmask_pos] = gt_token_id
    #                     unmasked_positions[seq_idx].add(next_unmask_pos)

    #             # After finishing this block, save decode orders and move to next block
    #             for seq_idx in active_in_this_block:
    #                 all_decode_orders[seq_idx].append(decode_orders[seq_idx])
    #                 current_block_idx[seq_idx] += 1

    #                 # Check if this sequence is done
    #                 if current_block_idx[seq_idx] >= num_blocks_per_seq[seq_idx]:
    #                     sequence_active[seq_idx] = False

    #     # Compute perplexity for each sequence
    #     results = []
    #     for seq_idx in range(batch_size):
    #         if len(all_log_probs[seq_idx]) > 0:
    #             nll = -np.mean(all_log_probs[seq_idx])
    #         else:
    #             perplexity = float("inf")

    #         results.append((nll, all_decode_orders[seq_idx]))

    #     return results