from ..constants import *
import torch, os, pickle, io, numpy as np

# some constants for this script

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def cpu_unpickle(fp):
    with open(fp, mode='rb') as f: return CPU_Unpickler(f).load()

def get_epoch(task, epoch, base):
    return torch.load(os.path.join(base, task, 'model.epoch-%d' % epoch), map_location='cpu')

def get_epoch_direct(d, epoch):
    return torch.load(os.path.join(d, 'model.epoch-%d' % epoch), map_location='cpu')

def validate_multilabel_weights(
    difference_w, multilabel_indices, multilabel_order=TASK_BINARY_MULTILABEL_ORDER
):
    # detecting if bias
    if difference_w.shape == (len(TASK_BINARY_MULTILABEL_ORDER),):
        col_w = difference_w
    else:
        col_w = difference_w.max(axis=1)

    for i in range(len(col_w)):
        if i in multilabel_indices:
            assert col_w[i] != 0, f"{multilabel_order[i]} weight doesn't change but should!"
        else:
            assert col_w[i] == 0, f"{multilabel_order[i]} weight isn't 0.  It should be 0"

def validate_multilabel_weights_task_gen(difference_w, multilabel_indices):
    # detecting if bias
    if difference_w.shape == (len(TASK_BINARY_MULTILABEL_ORDER),):
        col_w = difference_w
    else:
        col_w = difference_w.max(axis=1)

    for i in range(len(col_w)):
        if i in multilabel_indices:
            assert col_w[i] == 0, "%s weight changes! It isn't expected to change" % TASK_BINARY_MULTILABEL_ORDER[i]
        else:
            assert col_w[i] != 0, "%s weight is 0.  It shouldnt be 0" % TASK_BINARY_MULTILABEL_ORDER[i]



def pickle_weights_validate(base, task, current_ablation, multilabel_indices):
    task_weights = cpu_unpickle(os.path.join(base, task, 'joint_weights.pkl'))
    for k in task_weights['task_weights']:
        if k in current_ablation or k == 'tasks_binary_multilabel':
            assert task_weights['task_weights'][k] == 1, "%s should be set to 1" % k
        else:
            assert task_weights['task_weights'][k] == 0, "%s should be set to 0" % k

def pickle_weights_validate_task_gen(base, task, current_ablation, multilabel_indices):
    task_weights = cpu_unpickle(os.path.join(base, task, 'joint_weights.pkl'))
    for k in task_weights['task_weights']:
        if k not in current_ablation or k == 'tasks_binary_multilabel':
            assert task_weights['task_weights'][k] == 1, "%s should be set to 1" % k
        else:
            assert task_weights['task_weights'][k] == 0, "%s should be set to 0" % k

    # multilabel matrix ablation
    multilabel_task_weights = task_weights['task_class_weights']['tasks_binary_multilabel']
    for i in range(len(TASK_BINARY_MULTILABEL_ORDER)):
        if i not in multilabel_indices:
            assert multilabel_task_weights[i] == 1, "%s should be ablated" % TASK_BINARY_MULTILABEL_ORDER[i]
        else:
            assert multilabel_task_weights[i] == 0, "%s should not be ablated" % TASK_BINARY_MULTILABEL_ORDER[i]

def validate_singleton_weights_gen(
    base, ablate=None, encoder_should_change=True, args=None, do_assert=False,
    tasks_binary_multilabel_order=TASK_BINARY_MULTILABEL_ORDER,
):
    assert args is not None
    assert ablate == args.ablate

    uses_weight_decay = args.do_weight_decay
    do_masked_imputation = args.do_masked_imputation
    do_eicu = args.do_eicu

    if uses_weight_decay:
        # Can't account for weight_decay here.
        return False, None, (None, None, None, None)

    valid_weights_should_change = []
    valid_weights_should_not_change = []
    invalid_weights_should_change_but_do_not = []
    invalid_weights_should_not_change_but_do = []

    def _validate_multilabel_weights_fn(difference_w, static_multilabel_indices, overall_key):
        # detecting if bias
        if difference_w.shape == (len(tasks_binary_multilabel_order),): delta_w = difference_w
        else: delta_w = difference_w.max(axis=1)

        for i in range(len(delta_w)):
            key = tasks_binary_multilabel_order[i]
            if i in static_multilabel_indices:
                if delta_w[i] != 0:
                    if do_assert: raise AssertionError(f"{key} weight shouldn't change and does!")
                    else: invalid_weights_should_not_change_but_do.append((overall_key, key, delta_w[i]))
                else: valid_weights_should_not_change.append((overall_key, key, delta_w[i]))
            else:
                if delta_w[i] == 0:
                    if do_assert: raise AssertionError(f"{key} weights should change and doesn't!")
                    else: invalid_weights_should_change_but_do_not.append((overall_key, key))
                else: valid_weights_should_change.append((overall_key, key))

    def weights_changed(ep_1, ep_2, key=''):
        difference_w = ep_1.detach().cpu().numpy() - ep_2.detach().cpu().numpy()
        return np.abs(difference_w).max() > 0, (difference_w, np.abs(difference_w).max())

    def assert_diff(ep_1, ep_2, should_change=True, key=''):
        changed, delta = weights_changed(ep_1, ep_2)

        if do_assert:
            if should_change: assert changed, f"{key} should change!" + str(delta)
            else: assert not changed, f"{key} should not change!" + str(delta)
        else:
            if should_change and changed: valid_weights_should_change.append((key, delta))
            elif should_change and not changed: invalid_weights_should_change_but_do_not.append(key)
            elif not should_change and changed: invalid_weights_should_not_change_but_do.append((key, delta))
            elif not should_change and not changed: valid_weights_should_not_change.append(key)

    if type(ablate) is str: ablate = [ablate]
    assert type(ablate) is list or type(ablate) is tuple, f"Ablation {ablate} is the wrong type!"

    # At this point, everything is ablating 'next_timepoint'
    ablations = ['next_timepoint']
    ablated_task_heads = [f"task_heads.next_timepoint.{e}" for e in ('weight', 'bias')]

    if not do_masked_imputation:
        ablations.append('masked_imputation')
        ablated_task_heads.extend([f"task_heads.masked_imputation.{e}" for e in ('weight', 'bias')])

    if do_eicu:
        ablations.append('FTS')
        ablated_task_heads.extend([
            'FTS_decoder.decoder.C_proj.bias',
            'FTS_decoder.decoder.C_proj.weight',
            'FTS_decoder.decoder.H_proj.bias',
            'FTS_decoder.decoder.H_proj.weight',
            'FTS_decoder.decoder.LSTM.bias_hh_l0',
            'FTS_decoder.decoder.LSTM.bias_ih_l0',
            'FTS_decoder.decoder.LSTM.weight_hh_l0',
            'FTS_decoder.decoder.LSTM.weight_ih_l0',
            'FTS_decoder.decoder.X_proj.bias',
            'FTS_decoder.decoder.X_proj.weight',
            'FTS_decoder.decoder.treatment_embeddings.weight',
            'FTS_decoder.predictor.classifier.bias',
            'FTS_decoder.predictor.classifier.weight',
            'treatment_embeddings.weight',
        ])

    for task in ablate:
        if task in ABLATION_GROUPS:
            ablations.extend(ABLATION_GROUPS[task])
            ablated_task_heads.extend(TASK_HEAD_MAPPING[task])
        elif task in ('next_timepoint_was_measured', 'next_timepoint_info'):
            ablations.append('next_timepoint_info')
            ablated_task_heads.extend(
                [f"task_heads.next_timepoint_was_measured.{e}" for e in ('weight', 'bias')]
            )
        elif task == 'next_timepoint':
            ablations.append('next_timepoint')
            ablated_task_heads.extend(
                [f"task_heads.next_timepoint.{e}" for e in ('weight', 'bias')]
            )
        elif task == 'masked_imputation':
            ablated_task_heads.extend([f"task_heads.masked_imputation.{e}" for e in ('weight', 'bias')])
        else: raise AssertionError(
            f"{task} invalid. Should be in {ABLATION_GROUPS.keys()} or next_timepoint_info..."
        )

    # set for fast lookup
    # col indices to change in binary_multilabel matrix
    static_multilabel_indices = set([
        i for i, t in enumerate(tasks_binary_multilabel_order) if t in ablations
    ])

    ep_1_w = get_epoch_direct(base, 1)
    ep_2_w = get_epoch_direct(base, 4)

    for projector in ('ts_projector', 'statics_projector'):
        assert ep_1_w[projector].keys() == ep_2_w[projector].keys()
        for k in ep_1_w[projector]:
            assert_diff(
                ep_1_w[projector][k], ep_2_w[projector][k], encoder_should_change, f"{projector}[{k}]"
            )

    assert ep_1_w['model'].keys() == ep_2_w['model'].keys()
    for k in ep_1_w['model']:
        ep_1, ep_2 = ep_1_w['model'][k], ep_2_w['model'][k]
        if k.startswith('bert') or k.startswith('gru') or k.startswith('fc_stack'):
            assert_diff(ep_1, ep_2, encoder_should_change, k)
        elif k == 'task_losses.tasks_binary_multilabel.BCE_LL.pos_weight':
            assert_diff(ep_1, ep_2, False, k)
        elif k in (
            'task_heads.tasks_binary_multilabel.weight',
            'task_heads.tasks_binary_multilabel.bias'
        ):
            _validate_multilabel_weights_fn(
                np.abs(ep_1.detach().cpu().numpy() - ep_2.detach().cpu().numpy()), static_multilabel_indices,
                k
            )
        else:
            assert_diff(ep_1, ep_2, k not in ablated_task_heads, k)

    any_invalid = (
        (len(invalid_weights_should_change_but_do_not) > 0)
        or (len(invalid_weights_should_not_change_but_do) > 0)
    )
    return True, any_invalid, (
        valid_weights_should_change,
        valid_weights_should_not_change,
        invalid_weights_should_change_but_do_not,
        invalid_weights_should_not_change_but_do,
    )

def validate_singleton_weights(base, task=None, encoder_should_change=True, args=None):
    # TODO(mmd): Update to support masked_imputation validation
    all_tasks_should_change = (task is None)

    uses_weight_decay = args is not None and args.do_weight_decay
    assert not uses_weight_decay, "Can't validate model that is using weight decay!"

    if uses_weight_decay: print("Accounting for weight decay.")

    def weights_changed(ep_1, ep_2, key=''):
        difference_w = ep_1.detach().cpu().numpy() - ep_2.detach().cpu().numpy()
        if uses_weight_decay:
            mag_delta = (ep_1.detach().cpu().numpy() ** 2).mean() - (ep_2.detach().cpu().numpy() ** 2).mean()
            return mag_delta <= 0, (difference_w, difference_w.min(), mag_delta)
            #return difference_w.min() <= 0, (difference_w, difference_w.min(), mag_delta)
        else:
            return np.abs(difference_w).max() > 0, (difference_w, np.abs(difference_w).max())

    def assert_diff(ep_1, ep_2, should_change=True, key=''):
        changed, delta = weights_changed(ep_1, ep_2)
        if should_change: assert changed, f"{key} should change!" + str(delta)
        else: assert not changed, f"{key} should not change!" + str(delta)

    if task is not None:
        if task in ABLATION_GROUPS:
            current_ablation = ABLATION_GROUPS[task]
            current_task_heads = TASK_HEAD_MAPPING[task]
        elif task == 'next_timepoint_was_measured':
            current_ablation = 'next_timepoint_info'
            current_task_heads = {f"task_heads.next_timepoint_was_measured.{e}" for e in ('weight', 'bias')}
        elif task == 'masked_imputation':
            current_ablation = ''
            current_task_heads = {f"task_heads.masked_imputation.{e}" for e in ('weight', 'bias')}
        else: raise AssertionError(f"{task} invalid. Should be in {ABLATION_GROUPS} or next_timepoint_was...")

        # set for fast lookup
        # col indices to change in binary_multilabel matrix
        multilabel_indices = set([
            i for i, t in enumerate(TASK_BINARY_MULTILABEL_ORDER) if t in current_ablation
        ])

    ep_1_w = get_epoch_direct(base, 1)
    ep_2_w = get_epoch_direct(base, 4)

    for projector in ('ts_projector', 'statics_projector'):
        for k in ep_1_w[projector]:
            assert_diff(
                ep_1_w[projector][k], ep_2_w[projector][k], encoder_should_change, f"{projector}[{k}]"
            )

    for k in ep_1_w['model']:
        ep_1, ep_2 = ep_1_w['model'][k], ep_2_w['model'][k]
        if k.startswith('bert') or k.startswith('gru') or k.startswith('fc_stack'):
            assert_diff(ep_1, ep_2, encoder_should_change, k)
        elif all_tasks_should_change:
            assert_diff(ep_1, ep_2, True, k)
        elif k in (
            'task_heads.tasks_binary_multilabel.weight',
            'task_heads.tasks_binary_multilabel.bias'
        ):
            validate_multilabel_weights(
                np.abs(ep_1.detach().cpu().numpy() - ep_2.detach().cpu().numpy()), multilabel_indices
            )
        else:
            if k in current_task_heads: print("validating should change")
            assert_diff(ep_1, ep_2, k in current_task_heads, k)

def validate_weights(task, base, encoder_should_change=True):
    assert task in ABLATION_GROUPS, "task doesn't exist : not in ABLATION_GROUPS constant"
    current_ablation = ABLATION_GROUPS[task]
    current_task_heads = TASK_HEAD_MAPPING[task]
    # col indices to change in binary_multilabel matrix
    # set for fast lookup
    multilabel_indices = set([i for i, t in enumerate(TASK_BINARY_MULTILABEL_ORDER) if t in current_ablation])

    ep_1_w = get_epoch(task, 1, base)
    ep_22_w = get_epoch(task, 16, base)

    for projector in ('ts_projector', 'statics_projector'):
        for k in ep_1_w[projector]:
            difference_w = np.abs(
                    ep_1_w[projector][k].detach().cpu().numpy()
                    - ep_22_w[projector][k].detach().cpu().numpy()
            )
            if encoder_should_change:
                assert difference_w.max() != 0, f"{projector}[{k}] should change for {base}/{task}"
            else:
                assert difference_w.max() == 0, f"{projector}[{k}] shouldn't change for {base}/{task}"

    for k in ep_1_w['model']:
        difference_w = np.abs(
                ep_1_w['model'][k].detach().cpu().numpy()
                - ep_22_w['model'][k].detach().cpu().numpy()
        )
        if (
            k.startswith('bert') or k.startswith('gru') or k.startswith('fc_stack')
        ):
            if encoder_should_change:
                assert difference_w.max() != 0, f"{k}'s weights should change for {base}/{task}"
            else:
                assert difference_w.max() == 0, f"{k}'s weights shouldn't change for {base}/{task}"
        elif k in (
            'task_heads.tasks_binary_multilabel.weight',
            'task_heads.tasks_binary_multilabel.bias'
        ):
            validate_multilabel_weights(difference_w, multilabel_indices)
        elif k in current_task_heads:
            assert difference_w.max() != 0, f"{k} difference is 0 for {base}/{task}"
        else:
            assert difference_w.max() == 0, f"{k} difference is not 0 for {base}/{task}"

    # pickle weights validation
    pickle_weights_validate(base, task, current_ablation, multilabel_indices)

def validate_weights_task_gen(task, base):
    assert task in ABLATION_GROUPS, "task doesn't exist : not in ABLATION_GROUPS constant"
    current_ablation = ABLATION_GROUPS[task]
    current_task_heads = TASK_HEAD_MAPPING[task]
    # col indices to change in binary_multilabel matrix
    # set for fast lookup
    multilabel_indices = set([i for i, t in enumerate(TASK_BINARY_MULTILABEL_ORDER) if t in current_ablation])

    ep_1_w = get_epoch(task, 1, base)
    ep_22_w = get_epoch(task, 16, base)

    for projector in ('ts_projector', 'statics_projector'):
        for k in ep_1_w[projector]:
            difference_w = np.abs(
                    ep_1_w[projector][k].detach().cpu().numpy()
                    - ep_22_w[projector][k].detach().cpu().numpy()
            )
            assert difference_w.max() != 0, f"{projector}[{k}] should change for {base}/{task}"

    for k in ep_1_w['model']:
        difference_w = np.abs(
                ep_1_w['model'][k].detach().cpu().numpy()
                - ep_22_w['model'][k].detach().cpu().numpy()
        )
        if k.startswith('bert') or k.startswith('gru') or k.startswith('fc_stack'):
            assert difference_w.max() != 0, f"{k}'s weights should change"
        elif k == 'task_heads.tasks_binary_multilabel.weight' or k == 'task_heads.tasks_binary_multilabel.bias':
            validate_multilabel_weights_task_gen(difference_w, multilabel_indices)
        elif k in current_task_heads or k in ALWAYS_EQ_KEYS:
            assert difference_w.max() == 0, '%s difference is not 0' % k
        else:
            assert difference_w.max() != 0, '%s difference is 0' % k

    # pickle weights validation
    pickle_weights_validate_task_gen(base, task, current_ablation, multilabel_indices)

def validate_all(base, single_task=False, ft_encoder=False, tasks=None):
    if tasks is None: tasks = list(ABLATION_GROUPS.keys())
    assert os.path.isdir(base)
    tasks = set(tasks).intersection(os.listdir(base))
    print(f"validating for path: {base} and tasks {', '.join(tasks)}")
    for t in tasks:
        print(f"Validating {t}")
        if single_task:
            validate_weights(t, os.path.join(base, t), encoder_should_change=True)
        else:
            if not ft_encoder: validate_weights_task_gen(t, base)
            validate_weights(t, os.path.join(base, t), encoder_should_change = ft_encoder)
