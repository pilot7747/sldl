def patch_realesrgan_param_names(state_dict):
    state_dict = state_dict['params_ema']
    keys = list(state_dict.keys())

    for key in keys:
        new_key_name = None
        if key.startswith('body.'):
            new_key_name = key.replace('body.', 'RRDB_trunk.').replace('rdb', 'RDB')
        if key.startswith('conv_body.'):
            new_key_name = key.replace('conv_body.', 'trunk_conv.')
        if key.startswith('conv_up'):
            new_key_name = key.replace('conv_up', 'upconv')
        if key.startswith('conv_hr'):
            new_key_name = key.replace('conv_hr.', 'HRconv.')

        if new_key_name is not None:
            state_dict[new_key_name] = state_dict.pop(key)
    return state_dict
