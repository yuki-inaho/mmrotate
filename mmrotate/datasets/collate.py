# mmrotate/mmrotate/datasets/collate.py
from typing import List, Dict, Any
import torch
import collections
from mmengine.registry import FUNCTIONS # MMEngine の汎用関数レジストリ
from torch.utils.data._utils.collate import default_collate # PyTorchのデフォルトcollate


@FUNCTIONS.register_module() # レジストリに登録
def rgbd_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for RGBD data.

    This function handles a batch of data samples, specifically ensuring that
    the 'inputs' field (expected to be image tensors) is correctly stacked
    into a single batch tensor. Other fields in the samples are collated
    using PyTorch's `default_collate`.

    Args:
        batch (List[Dict[str, Any]]): A list of data samples, where each
            sample is a dictionary. Each dictionary is expected to have
            an 'inputs' keyHolding a tensor of shape (C, H, W) and
            a 'data_samples' key holding a DetDataSample object.

    Returns:
        Dict[str, Any]: A dictionary representing the collated batch.
        It will contain:
            - 'inputs': A stacked tensor of shape (N, C, H, W).
            - 'data_samples': A list of DetDataSample objects.
            - Other fields collated by `default_collate`.
    """
    collated_dict = {}
    
    # 'inputs' と 'data_samples' を特別に処理するために一時的に分離
    inputs_list = []
    data_samples_list = []
    other_fields_batch = [] # inputs と data_samples 以外のキーを持つ辞書のリスト

    for sample in batch:
        # sample は {'inputs': Tensor, 'data_samples': DetDataSample, (optionally_other_keys)...}
        current_sample_other_fields = {}
        if 'inputs' in sample:
            inputs_list.append(sample['inputs'])
        else:
            # 'inputs' がない場合はエラーまたは警告を出すべき
            raise KeyError("Sample in batch is missing 'inputs' key.")
            
        if 'data_samples' in sample:
            data_samples_list.append(sample['data_samples'])
        else:
            raise KeyError("Sample in batch is missing 'data_samples' key.")
            
        for key, value in sample.items():
            if key not in ['inputs', 'data_samples']:
                current_sample_other_fields[key] = value
        if current_sample_other_fields: # 他のキーがあれば追加
             other_fields_batch.append(current_sample_other_fields)

    # 'inputs' をスタック
    if inputs_list:
        collated_dict['inputs'] = torch.stack(inputs_list, 0)
    
    # 'data_samples' はリストのまま
    if data_samples_list:
        collated_dict['data_samples'] = data_samples_list
        
    # inputs と data_samples 以外のフィールドがあれば、それらを default_collate で処理
    if other_fields_batch:
        try:
            collated_dict.update(default_collate(other_fields_batch))
        except TypeError as e:
            # default_collate が処理できない型が含まれている場合
            # (例: BaseBoxes のリストなど、特殊な処理が必要な場合)
            # ここでは単純にエラーを出すか、キーごとに個別の処理を検討する
            print(f"Warning: default_collate failed for other_fields_batch: {e}. Other fields might not be collated correctly.")
            # 代わりに、各キーを手動でリスト化するなどのフォールバックも考えられる
            # for key in other_fields_batch[0].keys():
            #    collated_dict[key] = [d[key] for d in other_fields_batch]
            pass # またはエラーを raise

    return collated_dict