"""Models, leave-one-run-out evaluation, temperature scaling and metrics for the judgement POC (CPU torch)."""
import numpy as np
import torch

torch.set_num_threads(4)


def auroc(scores, y):
    scores, y = np.asarray(scores, float), np.asarray(y, int)
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), float)
    s = scores[order]
    i = 0
    while i < len(s):  # average ranks for ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float('nan')
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    total, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum():
            conf, acc = p[m].mean(), y[m].mean()
            total += m.sum() / len(p) * abs(acc - conf)
            table.append(dict(bin=f'{lo:.1f}-{hi:.1f}', n=int(m.sum()), mean_p=round(float(conf), 4), frac_pos=round(float(acc), 4)))
    return float(total), table


def metrics(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    eps = 1e-7
    e, table = ece(p, y)
    return dict(auroc=round(auroc(p, y), 4), ece=round(e, 4), brier=round(float(np.mean((p - y) ** 2)), 4),
                logloss=round(float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))), 4),
                n=int(len(y)), positives=int(y.sum()), reliability=table)


def bootstrap_auroc(p, y, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx))
        vals.append(auroc(p[b], y[b]))
    return [round(float(np.nanpercentile(vals, 2.5)), 4), round(float(np.nanpercentile(vals, 97.5)), 4)]


class Net(torch.nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        if hidden:
            self.f = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                         torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                         torch.nn.Linear(hidden, 1))
        else:
            self.f = torch.nn.Linear(d, 1)

    def forward(self, x):
        return self.f(x).squeeze(-1)


def fit(X, y, hidden, epochs, lr, wd, X_val=None, seed=0):
    """Full-batch Adam on BCE. Returns the model and, if X_val is given, the validation logits after every epoch."""
    torch.manual_seed(seed)
    net = Net(X.shape[1], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.float32)
    Xv = torch.tensor(X_val) if X_val is not None else None
    val_logits = []
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt), yt)
        loss.backward()
        opt.step()
        if Xv is not None:
            net.eval()
            with torch.no_grad():
                val_logits.append(net(Xv).numpy().copy())
    net.eval()
    return net, (np.stack(val_logits) if Xv is not None else None)


def fit_temperature(logits, y):
    """Temperature T > 0 minimising BCE of sigmoid(logit / T)."""
    log_t = torch.zeros(1, requires_grad=True)
    z, yt = torch.tensor(logits, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z / torch.exp(log_t), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(log_t).item())


def standardize(X_train, X_other):
    mu, sd = X_train.mean(0), X_train.std(0)
    sd[sd < 1e-6] = 1.0
    return (X_train - mu) / sd, (X_other - mu) / sd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def leave_one_run_out(X, y, groups, hidden=0, epochs=300, lr=None, wd=1e-3, seed=0, log=print):
    """Outer: hold out each run. Inner: leave-one-run-out over the training runs gives out-of-fold logits at every
    epoch, which pick the epoch count (lowest mean BCE) and fit the temperature. The final model is trained on all
    training runs for that many epochs and its test logits are divided by the temperature."""
    lr = lr if lr is not None else (0.05 if hidden == 0 else 0.003)
    runs = sorted(set(groups))
    p_cal, p_raw, fold_info = np.zeros(len(y)), np.zeros(len(y)), []
    for test_run in runs:
        test = groups == test_run
        train_runs = [r for r in runs if r != test_run]
        # inner out-of-fold logits per epoch
        oof = np.zeros((epochs, int((~test).sum())))
        train_idx = np.where(~test)[0]
        pos_in_train = {i: k for k, i in enumerate(train_idx)}
        for val_run in train_runs:
            val = (groups == val_run)
            fit_mask = ~test & ~val
            Xf, Xv = standardize(X[fit_mask], X[val])
            _, vl = fit(Xf, y[fit_mask], hidden, epochs, lr, wd, Xv, seed)
            cols = [pos_in_train[i] for i in np.where(val)[0]]
            oof[:, cols] = vl
        y_train = y[~test]
        bce = [float(np.mean(np.logaddexp(0, -oof[e]) * y_train + np.logaddexp(0, oof[e]) * (1 - y_train))) for e in range(epochs)]
        best = int(np.argmin(bce)) + 1
        T = fit_temperature(oof[best - 1], y_train)
        Xf, Xt = standardize(X[~test], X[test])
        net, _ = fit(Xf, y[~test], hidden, best, lr, wd, None, seed)
        with torch.no_grad():
            z = net(torch.tensor(Xt)).numpy()
        p_raw[test], p_cal[test] = sigmoid(z), sigmoid(z / T)
        fold_info.append(dict(run=test_run, n=int(test.sum()), positives=int(y[test].sum()), epochs=best, temperature=round(T, 3),
                              auroc=round(auroc(p_cal[test], y[test]), 4) if y[test].sum() and (1 - y[test]).sum() else None))
        log(f'  fold {test_run[-22:]}: n={int(test.sum())} pos={int(y[test].sum())} epochs={best} T={T:.2f} auroc={fold_info[-1]["auroc"]}')
    return p_cal, p_raw, fold_info
