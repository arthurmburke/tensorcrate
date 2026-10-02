import time, torch, torch.nn.functional as F

assert torch.backends.mps.is_available()
dev = "mps"
sync = torch.mps.synchronize


def bench(fn, reps_lat=200, reps_pipe=64, rounds=30):
    for _ in range(20):
        fn()
    sync()
    lat = []
    for _ in range(reps_lat):
        t = time.perf_counter()
        fn()
        sync()
        lat.append(time.perf_counter() - t)
    pipe = []
    for _ in range(rounds):
        t = time.perf_counter()
        for _ in range(reps_pipe):
            fn()
        sync()
        pipe.append((time.perf_counter() - t) / reps_pipe)
    lat.sort()
    pipe.sort()
    return lat[len(lat) // 2] * 1e6, pipe[len(pipe) // 2] * 1e6  # µs


def gelu_prim(x):
    return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def row(name, fn):
    l, p = bench(fn)
    print(f"{name:<44}{l:>12.3f} µs (lat){p:>12.3f} µs (pipe)")


for n in (4096, 65536, 1048576, 16777216):
    x = torch.randn(n, device=dev)
    row(f"relu n={n}", lambda: F.relu(x))
    row(f"sigmoid n={n}", lambda: torch.sigmoid(x))
    row(f"gelu (tanh, single kernel) n={n}", lambda: F.gelu(x, approximate="tanh"))
    row(f"gelu (primitives, ~9 ops) n={n}", lambda: gelu_prim(x))

for r, c in ((64, 256), (256, 1024), (1024, 1024)):
    x = torch.randn(r, c, device=dev)
    row(f"layer_norm {r}x{c}", lambda: F.layer_norm(x, (c,)))
for cls, s in ((128, 128), (1024, 256), (4096, 512)):
    x = torch.randn(s, cls, device=dev)
    row(f"softmax {cls} classes x {s} samples", lambda: F.softmax(x, dim=-1))

for dt in (torch.float32, torch.float16):
    for n in (256, 512, 1024, 2048):
        a = torch.randn(n, n, device=dev, dtype=dt)
        b = torch.randn(n, n, device=dev, dtype=dt)
        l, p = bench(lambda: a @ b)
        print(
            f"matmul {dt} {n}^3: {2*n**3/(p*1e-6)/1e9:>9.1f} GFLOP/s (pipe), lat {l:.1f} µs"
        )

for m, k, n in ((32, 256, 256), (128, 512, 512), (512, 1024, 1024)):
    X = torch.randn(m, k, device=dev)
    W = torch.randn(k, n, device=dev)
    bb = torch.randn(n, device=dev)
    l, p = bench(lambda: F.relu(X @ W + bb))
    print(f"dense relu {m}x{k}x{n}: {2*m*k*n/(p*1e-6)/1e9:>9.1f} GFLOP/s (pipe)")
    try:
        cf = torch.compile(lambda: F.relu(X @ W + bb))
        l, p = bench(cf)
        print(f"  compiled: {2*m*k*n/(p*1e-6)/1e9:>9.1f} GFLOP/s (pipe)")
    except Exception as e:
        print("  torch.compile failed:", type(e).__name__)

n = 1048576
for fused in (False, True):
    p_ = torch.nn.Parameter(torch.randn(n, device=dev))
    p_.grad = torch.randn(n, device=dev)
    try:
        opt = torch.optim.Adam([p_], lr=1e-3, fused=fused)
        row(f"Adam n={n} fused={fused}", opt.step)
    except Exception as e:
        print(f"Adam fused={fused} failed:", type(e).__name__)

for din, dh, dout, bs in ((64, 256, 16, 128), (256, 1024, 64, 256)):
    net = torch.nn.Sequential(
        torch.nn.Linear(din, dh), torch.nn.ReLU(), torch.nn.Linear(dh, dout)
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    X = torch.randn(bs, din, device=dev)
    Y = torch.randn(bs, dout, device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        F.mse_loss(net(X), Y).backward()
        opt.step()

    l, p = bench(step)
    print(
        f"MLP {din}->{dh}->{dout} b{bs}: {1e6/l:.0f} steps/s (lat), {1e6/p:.0f} steps/s (pipe)"
    )
