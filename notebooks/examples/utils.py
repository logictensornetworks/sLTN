from ltlf2dfa.ltlf import LTLfAtomic, LTLfNot, LTLfAnd, LTLfOr, LTLfImplies, LTLfAlways, LTLfEventually, LTLfUntil, \
    LTLfNext, LTLfEquivalence, LTLfFormula
from ltlf2dfa.parser.ltlf import LTLfParser
import numpy as np
from pysat.formula import CNFPlus
from pysat.solvers import Solver
import torch
from tqdm import tqdm

def ltl_to_fol(formula_str, variable, dimension):
    parser = LTLfParser()
    formula = parser(formula_str)
    return f"({_to_fol(formula,variable,dimension,depth=0)})[{dimension}=0]"

def _to_fol(formula, variable, dimension,depth):
    dim = dimension if depth == 0 else f"{dimension}{depth}"
    dim1 = f"{dimension}{depth+1}"
    dim2 = f"{dimension}{depth+2}"
    if type(formula) == LTLfAtomic:
        return f"{formula}({variable}[{dim}])"
    elif type(formula) == LTLfNot:
        return f"~({_to_fol(formula.f, variable, dimension, depth)})"
    elif type(formula) == LTLfAnd:
        return "(" + " and ".join([_to_fol(f, variable, dimension, depth) for f in formula.formulas]) + ")"
    elif type(formula) == LTLfOr:
        return "(" + " or ".join([_to_fol(f, variable, dimension, depth) for f in formula.formulas]) + ")"
    elif type(formula) == LTLfImplies:
        f,g = formula.formulas
        return f"({_to_fol(f, variable, dimension, depth)}) -> ({_to_fol(g,variable, dimension, depth)})"
    elif type(formula) == LTLfEquivalence:
        return "(" + "<->".join([_to_fol(f, variable, dimension, depth) for f in formula.formulas]) + ")"
    elif type(formula) == LTLfAlways:
        return f"(forall {dim1} | After({dim},{dim1}) : ({_to_fol(formula.f, variable, dimension, depth + 1)}))"
    elif type(formula) == LTLfEventually:
        return f"(exists {dim1} | After({dim},{dim1}) : ({_to_fol(formula.f, variable, dimension, depth + 1)}))"
    elif type(formula) == LTLfNext:
        return f"(exists {dim1} | Next({dim},{dim1}): ({_to_fol(formula.f, variable, dimension, depth + 1)}))"
    elif type(formula) == LTLfUntil:
        f,g = formula.formulas
        return "".join([
            f"exists {dim1} | After({dim},{dim1}) : ( {_to_fol(g,variable,dimension, depth + 1)} & ",
            f"forall {dim2} | After({dim},{dim2}) & After({dim2},{dim1}) : ({_to_fol(g,variable,dimension, depth+2)} or {_to_fol(f,variable,dimension, depth+2)}))"])
    else:
        return None

def ltl_to_cnf(ltlf_f,l,tz_vars):
    cnf = CNFPlus()
    _ = _to_cnf(ltlf_f,0,l,cnf,tz_vars)
    return cnf

def _to_cnf(ltlf_f, d, l, cnf, tz_vars):

    if isinstance(ltlf_f,LTLfAtomic):
        return _get_tz_var(ltlf_f, d, tz_vars)

    elif isinstance(ltlf_f, LTLfNot):
        f = ltlf_f.f
        var_f = _to_cnf(f, d, l, cnf, tz_vars)
        var = _get_tz_var(ltlf_f, d, tz_vars)
        cnf.append([var, var_f])
        cnf.append([-var, -var_f])
        return var

    elif isinstance(ltlf_f,LTLfAnd):
        ff = ltlf_f.formulas
        var_ff = [_to_cnf(f, d, l, cnf, tz_vars) for f in ff]
        var = _get_tz_var(ltlf_f, d, tz_vars)
        cnf.append([-v for v in var_ff] + [var])
        for v in var_ff:
            cnf.append([-var,v])
        return var

    elif isinstance(ltlf_f,LTLfOr):
        ff = ltlf_f.formulas
        var_ff = [_to_cnf(f, d, l, cnf, tz_vars) for f in ff]
        var = _get_tz_var(ltlf_f, d, tz_vars)
        cnf.append(var_ff + [-var])
        for v in var_ff:
            cnf.append([var, -v])
        return var

    elif type(ltlf_f) == LTLfImplies:
        f,g= ltlf_f.formulas
        var_f,var_g = [_to_cnf(h, d, l, cnf, tz_vars) for h in [f, g]]
        var = _get_tz_var(ltlf_f, d, tz_vars)
        cnf.append([-var,-var_f,var_g])
        cnf.append([var, var_f])
        cnf.append([var, -var_g])
        return var

    elif type(ltlf_f) ==LTLfEquivalence:
        f, g = ltlf_f.formulas
        var_f, var_g = [_to_cnf(h, d, l, cnf, tz_vars) for h in [f, g]]
        var = _get_tz_var(ltlf_f, d, tz_vars)
        cnf.append([-var, -var_f, var_g])
        cnf.append([-var, -var_g, var_f])
        cnf.append([var, var_f,var_g])
        cnf.append([var, -var_f,-var_g])
        return var

    elif type(ltlf_f) == LTLfAlways:
        f = ltlf_f.f
        var_Gf = [_to_cnf(f, dl, l, cnf, tz_vars) for dl in range(d, l)]
        var = _get_tz_var(ltlf_f, d, tz_vars)
        cnf.append([-v for v in var_Gf] + [var])
        for v in var_Gf:
            cnf.append([-var, v])
        return var

    elif type(ltlf_f) == LTLfEventually:
        f = ltlf_f.f
        var_Ff = [_to_cnf(f, dl, l, cnf, tz_vars) for dl in range(d, l)]
        var = _get_tz_var(ltlf_f, d, tz_vars)
        cnf.append([v for v in var_Ff] + [-var])
        for v in var_Ff:
            cnf.append([var, -v])
        return var

    elif type(ltlf_f) == LTLfNext:
        f = ltlf_f.f
        var = _get_tz_var(ltlf_f, d, tz_vars)
        if d == l-1:
            cnf.append([-var])
        else:
            var_f = _to_cnf(f, d + 1, l, cnf, tz_vars)
            cnf.append([-var_f,var])
            cnf.append([-var,var_f])
        return var

    elif type(ltlf_f) == LTLfUntil:
        f,g = ltlf_f.formulas
        if d == l-1:
            var = _to_cnf(g, d, l, cnf, tz_vars)
        else:
            var = _to_cnf(LTLfOr([g, LTLfAnd([f, LTLfNext(ltlf_f)])]), d, l, cnf, tz_vars)
        var1 = _get_tz_var(ltlf_f, d, tz_vars)
        cnf.append([-var1, var])
        cnf.append([-var,var1])
        return var

    else:
        raise NotImplementedError(f"Formula type {type(ltlf_f)} not supported.")

def _get_tz_var(ltlf_f, d, tz_vars):
    assert isinstance(ltlf_f,LTLfFormula)
    var_label = f"x_{ltlf_f}_{d}"
    if var_label not in tz_vars:
        tz_vars[var_label] = len(tz_vars) + 1
    return tz_vars[var_label]

def get_models_for_cnf(cnf_f,n):
    models = []
    while len(models) < n:
        with Solver(name="Minicard",bootstrap_with=cnf_f) as solver:
            if solver.solve():
                for m in solver.enum_models():
                    select_m = np.random.choice([0,1],p=(0.99,0.01))
                    if select_m == 1:
                        models.append(m)
                        if len(models) >= n:
                            break
            else:
                return []
    return models


def generate_data_sequence_dataset_for_ltlf(
        ltlf_f_strings,
        len_traces,
        nr_of_samples,
        dataset,
        atom_to_class,
        atleast=1,
        atmost=1):

    data_shape = dataset.data[0].shape
    data = np.zeros((nr_of_samples * 2 * len(ltlf_f_strings), len_traces) + data_shape)
    labels = np.zeros((len(ltlf_f_strings) * 2 * nr_of_samples, len(ltlf_f_strings)))
    parser = LTLfParser()
    ltlf_ff = [parser(ltlf_f_str) for ltlf_f_str in ltlf_f_strings]

    tz_vars = dict()
    tz_vars_atoms = {d:dict() for d in range(len_traces)}
    for d in range(len_traces):
        for a in atom_to_class:
            tz_vars_atoms[d][parser(a)] = _get_tz_var(parser(a),d,tz_vars)
    atl_atm_cnf = CNFPlus()
    if atmost is not None:
        for d in range(len_traces):
            atl_atm_cnf.append([list(tz_vars_atoms[d].values()), atmost], is_atmost=True)
    if atleast is not None:
        for d in range(len_traces):
            atl_atm_cnf.append([[-v for v in list(tz_vars_atoms[d].values())], len(tz_vars_atoms[d]) - atleast], is_atmost=True)

    cnf_ff = []

    for idx_f,ltlf_f in enumerate(ltlf_ff):
        cnf_f = ltl_to_cnf(ltlf_f,len_traces,tz_vars)
        cnf_f.extend(atl_atm_cnf)
        cnf_ff.append(cnf_f)

    for idx_f, ltlf_f in tqdm(enumerate(ltlf_ff)):
        for v in range(2):
            cnf_f = cnf_ff[idx_f]
            cnf_fv = CNFPlus()
            cnf_fv.clauses = [c for c in cnf_f.clauses]
            cnf_fv.atmosts = [atm for atm in cnf_f.atmosts]
            cnf_fv.append([(-1 + 2 * v) * _get_tz_var(ltlf_f, 0, tz_vars)])
            models = get_models_for_cnf(cnf_fv,nr_of_samples)

            for m, model in enumerate(models):
                sample_idx = nr_of_samples*(2*idx_f + v) + m

                for d in range(len_traces):
                    for a in atom_to_class:
                        tz_var_a_at_d = _get_tz_var(parser(a),d,tz_vars)
                        if model[tz_var_a_at_d - 1] > 0:
                            data[sample_idx,d] = dataset.data[np.random.choice(np.where(dataset.targets == atom_to_class[a])[0])]

                assumptions = [model[v - 1] for d in range(len_traces) for v in tz_vars_atoms[d].values()]
                for idx_f1,ltlf_f1 in enumerate(ltlf_ff):
                    with Solver(name="Minicard", bootstrap_with=cnf_ff[idx_f1]) as solver:
                        if solver.solve(assumptions=assumptions):
                            mf = solver.get_model()
                            labels[sample_idx,idx_f1] = int(mf[_get_tz_var(ltlf_f1,0,tz_vars)-1] > 0)

    return torch.tensor(data).to(torch.float32).unsqueeze(-3), torch.tensor(labels).to(torch.float32)


def ground_formula_on_atoms(f_str, atoms):
    ground_f_str = []
    for i in range(len(atoms) - 1):
        replacement = str.maketrans({"a":atoms[i], "b":atoms[i + 1]})
        ground_f_str.append(f"({f_str.translate(replacement)})")
    if len(atoms) == 1:
        return ground_f_str[0]
    else:
        return " & ".join(ground_f_str)

import torch.nn as nn

class MNISTCNN(nn.Module):
    def __init__(self,nr_of_classes):
        super(MNISTCNN, self).__init__()

        self.features = nn.Sequential(
            # Input: (1, 28, 28)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),          # (32, 14, 14)
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),          # (64, 7, 7)
            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, nr_of_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class DigitCNN(torch.nn.Module):
    """Simple CNN for digit classification."""

    def __init__(self,nr_of_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 7 * 7, 64)
        self.fc2 = torch.nn.Linear(64, nr_of_classes)

    def forward(self, x):
        x = x.unsqueeze(-3)  # Add channel dimension

        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim=-3)
        x = torch.relu(self.fc1(x))

        x = torch.softmax(self.fc2(x), dim=-1)

        return x.unsqueeze(-1)  # Add dimension for output_dims=["d"]

class Project(torch.nn.Module):

    def __init__(self,class_id):
        super().__init__()
        self.class_id = class_id

    def forward(self,x):
        x = x.select(-1,self.class_id)
        x = x.unsqueeze(-1)
        return x


class DataLoader:
    def __init__(self, data, labels, batch_size=64, shuffle=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        idxlist = list(range(self.data.shape[0]))
        if self.shuffle:
            np.random.shuffle(idxlist)
        for start in range(0, self.data.shape[0], self.batch_size):
            idx = idxlist[start:start + self.batch_size]
            yield self.data[idx], self.labels[idx]
