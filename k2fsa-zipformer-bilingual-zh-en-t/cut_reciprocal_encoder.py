import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np

# 1. 读取模型
# python dir/cut_reciprocal_encoder.py \
#   --onnx_ori $dir_onnx/encoder-epoch-99-avg-1.sim.onnx \
#   --onnx_new $dir_onnx/encoder-epoch-99-avg-1.sim.onnx
import argparse
parser = argparse.ArgumentParser(description="Replace Reciprocal nodes with Div nodes in an ONNX model")
parser.add_argument("--onnx_ori", type=str, required=True, help="Path to the original ONNX model")
parser.add_argument("--onnx_new", type=str, required=True, help="Path to the new ONNX model")
args = parser.parse_args()

def __main__():

    model_path = args.onnx_ori
    model = onnx.load(model_path)
    graph = model.graph

    new_nodes = []

    # 2. 遍历图中的节点
    for node in graph.node:
        if node.op_type == "Reciprocal":
            # node.input[0] 是 Reciprocal 的输入
            # node.output[0] 是 Reciprocal 的输出

            # 2.1 创建常量 1（标量）
            const_name = node.name + "_one"

            # 注意 dtype 要和输入一致，这里假设是 float32
            const_tensor = numpy_helper.from_array(
                np.array(1.0, dtype=np.float32),
                name=const_name
            )
            graph.initializer.append(const_tensor)

            # Constant 节点（某些工具会把常量直接从 initializer 读取，
            # 如果你的后端不要求 Constant 节点，可以不建 Constant 节点，只保留 initializer）
            const_node = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[const_name + "_out"],
                value=const_tensor,
                name=const_name + "_const"
            )

            # 2.2 创建 Div 节点：1 / x
            # 输入：const_node 的输出、原 Reciprocal 的输入
            # 输出：沿用原 Reciprocal 的输出名，保证后续计算图不变
            div_node = helper.make_node(
                "Div",
                inputs=[const_name + "_out", node.input[0]],
                outputs=node.output,
                name=node.name + "_div"
            )

            new_nodes.append(const_node)
            new_nodes.append(div_node)

        else:
            # 非 Reciprocal 节点直接保留
            new_nodes.append(node)

    # 3. 用新的节点列表替换原图中的节点
    del graph.node[:]
    graph.node.extend(new_nodes)

    # 4. 保存新模型
    onnx.save(model, args.onnx_new)
    print(f"Saved to {args.onnx_new}")
