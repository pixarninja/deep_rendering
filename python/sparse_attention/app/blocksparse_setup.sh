#!/bin/bash

sed 's/f"P:{P} Q:{Q} H:{H} W:{W} R:{R} S:{S} std:{stride} pad_r:{pad_r} pad_s:{pad_s}"/"P:{} Q:{} H:{} W:{} R:{} S:{} std:{} pad_r:{} pad_s:{}".format(P, Q, H, W, R, S, stride, pad_r, pad_s)/g' /usr/local/lib/python3.5/dist-packages/blocksparse/utils.py > temp_out
mv temp_out temp
sed 's/f"P:{P} Q:{Q}"/"P:{} Q:{}".format(P, Q)/g' temp > temp_out
mv temp_out temp
sed 's/f"H:{H} W:{W}"/"H:{} W:{}".format(H, W)/g' temp > temp_out
mv temp_out /usr/local/lib/python3.5/dist-packages/blocksparse/utils.py
sed 's/f"incompatible mask_shape: {mask_shape} x.shape: {x.shape}"/"incompatible mask_shape: {} x.shape: {}".format(mask_shape, x.shape)/g' /usr/local/lib/python3.5/dist-packages/blocksparse/ewops.py > temp_out
mv temp_out /usr/local/lib/python3.5/dist-packages/blocksparse/ewops.py
sed 's/"Incompatible shapes between op input {x.shape} and calculated input gradient {dx.shape} for {op.name} (idx:{i})")/"Incompatible shapes between op input {} and calculated input gradient {} for {} (idx:{})".format(x.shape, dx.shape, op.name, i))/g' /usr/local/lib/python3.5/dist-packages/blocksparse/grads.py > temp_out
mv temp_out temp
sed 's/f"Num gradients {len(dxs)} generated for op {op.node_def} do not match num inputs {len(op.inputs)}"/"Num gradients {} generated for op {} do not match num inputs {}".format(len(dxs), op.node_def, len(op.inputs))/g' temp > temp_out
mv temp_out temp
sed "s/f\"No gradient defined for operation '{op.name}' (op type: {op.type})\")/\"No gradient defined for operation '{}' (op type: {})\".format(op.name, op.type))/g" temp > temp_out
mv temp_out temp
sed 's/f"grad_ys_{i}"/"grad_ys_{}".format(i)/g' temp > temp_out
mv temp_out /usr/local/lib/python3.5/dist-packages/blocksparse/grads.py
sed 's/f"{name}_lut_{g_lut_idx}"/"{}_lut_{}".format(name, g_lut_idx)/g' /usr/local/lib/python3.5/dist-packages/blocksparse/matmul.py > temp_out
mv temp_out temp
sed 's/f"{scope}\/BlocksparseMatmulDG"/"{}\/BlocksparseMatmulDG".format(scope)/g' temp > temp_out
mv temp_out temp
sed 's/f"bad type: {addn_ops\[0\].type} Cause: this segment does not share a broadcasted gate."/"bad type: {} Cause: this segment does not share a broadcasted gate.".format(addn_ops\[0\].type)/g' temp > temp_out
mv temp_out temp
sed 's/f"splice failed for {dg_consumer.name}"/"splice failed for {}".format(dg_consumer.name)/g' temp > temp_out
mv temp_out /usr/local/lib/python3.5/dist-packages/blocksparse/matmul.py
