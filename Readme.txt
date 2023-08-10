0. 安装环境
pip install -r requirements.txt

1. 转换Llama原始模型权重为hf格式
如果是float16类型，使用convert_llama_weights_to_hf.py;model_size指定转换哪一个Llama模型
示例：
python convert_llama_weights_to_hf.py --input_dir /input2/ --model_size 7B --output_dir ../llama-7B-fp16/

如果是bfloat16类型，使用convert_llama_weights_to_hf_bfloat.py
示例：
python convert_llama_weights_to_hf.py --input_dir /input2/ --model_size 7B --output_dir ../llama-7B-bfp16/

2. 启动训练
训练float16类型，使用run.sh，根据服务器环境，在run.sh里指定显卡：--include localhost:0,1；在config/args_lomo.yaml里model_name_or_path指定步骤1的模型路径

训练bfloat16类型，则使用run_bf16.sh
