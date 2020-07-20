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

def validate_multilabel_weights(difference_w, multilabel_indices):
    # detecting if bias
    if difference_w.shape == (len(TASK_BINARY_MULTILABEL_ORDER),):
        col_w = difference_w
    else:
        col_w = difference_w.max(axis=1)

    for i in range(len(col_w)):
        if i in multilabel_indices:
            assert col_w[i] != 0, "%s weight doesn't change! It is expected to change" % TASK_BINARY_MULTILABEL_ORDER[i]
        else:
            assert col_w[i] == 0, "%s weight isn't 0.  It should be 0" % TASK_BINARY_MULTILABEL_ORDER[i]

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
