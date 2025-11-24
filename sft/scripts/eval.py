

from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='qwen_3_4b_instruct-turbomind',
        path='Qwen/Qwen3-4B-Instruct-2507',
        engine_config=dict(session_len=8196, max_batch_size=16, tp=1),
        gen_config=dict(
            temperature=0., top_p=1, do_sample=False
        ),
        max_seq_len=8192,
        max_out_len=8192,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    ),
]