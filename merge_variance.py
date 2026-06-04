import onnx
from pathlib import Path

base_dir = Path('D:/OpenUtau for diffsinger/Singers/SSM_test_opencpop/dsvariance')

# Load sub-models
var_pre = onnx.load(base_dir / 'var_testssm2.variance_pre.onnx')
var_diffusion = onnx.load(base_dir / 'var_testssm2.variance.onnx')
var_post = onnx.load(base_dir / 'var_testssm2.variance_post.onnx')

# Build merged graph manually
from onnx import helper

all_nodes = []
all_inits = []
all_inputs = []
all_outputs = []

# Process pre model
for inp in var_pre.graph.input:
    if inp.name not in ['variance_cond']:
        all_inputs.append(inp)
for init in var_pre.graph.initializer:
    all_inits.append(init)
for node in var_pre.graph.node:
    all_nodes.append(node)

# Process diffusion model
for inp in var_diffusion.graph.input:
    if inp.name != 'variance_cond':
        all_inputs.append(inp)
for init in var_diffusion.graph.initializer:
    all_inits.append(init)
for node in var_diffusion.graph.node:
    all_nodes.append(node)

# Process post model
for inp in var_post.graph.input:
    if inp.name != 'xs_pred':
        all_inputs.append(inp)
for init in var_post.graph.initializer:
    all_inits.append(init)
for node in var_post.graph.node:
    all_nodes.append(node)

# Collect outputs from post model
for out in var_post.graph.output:
    all_outputs.append(out)

# Create merged graph
merged_graph = helper.make_graph(
    nodes=all_nodes,
    name='merged_variance',
    inputs=all_inputs,
    outputs=all_outputs,
    initializer=all_inits,
)

# Create merged model
merged_model = helper.make_model(merged_graph, opset_imports=var_pre.opset_import)
merged_model.ir_version = var_pre.ir_version

# Save
output_path = base_dir / 'var_testssm2.variance.onnx'
onnx.save(merged_model, output_path)
print(f"Saved merged model to {output_path}")

# Verify
print("Inputs:", [i.name for i in merged_model.graph.input])
print("Outputs:", [o.name for o in merged_model.graph.output])
