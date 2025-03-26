import argparse
import sys
import tensorrt as trt
import os
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def build_engine(onnx_file_path, img_size, max_workspace_size=1 << 29):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(EXPLICIT_BATCH) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:
        
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
        config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUDNN))
        
        with open(onnx_file_path, "rb") as model_file:
            if not parser.parse(model_file.read()):
                print("❌ ONNX 파싱 실패!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        input_tensor = network.get_input(0)
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name,
                  (1, 3, img_size, img_size),  # MIN
                  (1, 3, img_size, img_size),  # OPT
                  (1, 3, img_size, img_size))  # MAX

        config.add_optimization_profile(profile)
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("❌ TensorRT 엔진 직렬화 실패!")
            return None

        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(serialized_engine)

def save_engine(engine, engine_file_path):
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, required=True, help="Input image size")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint")
    args = parser.parse_args()
    modelFile_path = f"outputs/models/lseg_img_enc_rn101_" + \
                    f"{os.path.basename(args.weights).split('_')[0]}_"+ \
                    f"{args.img_size}"
    
    onnx_path = modelFile_path + f".onnx"
    engine_path = modelFile_path + f".trt"
    
    engine = build_engine(onnx_path, args.img_size)
    if engine:
        save_engine(engine, engine_path)
        print(f"✅ TensorRT 엔진이 {engine_path} 로 저장되었습니다.")
    else:
        print("❌ TensorRT 변환에 실패했습니다.")
