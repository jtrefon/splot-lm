"""Float64 NumPy oracle for RFC 0003; not a production kernel or LM."""
import unittest
import numpy as np


def recurrent(q, k, v, alpha, beta, state):
    s = state.copy()
    out = []
    for qt, kt, vt, at, bt in zip(q, k, v, alpha, beta):
        s = at * s
        s = s + np.outer(kt, bt * (vt - kt @ s))
        out.append(qt @ s)
    return np.stack(out), s


def chunk(q, k, v, alpha, beta, state):
    """Independent dense triangular solve; positive alpha only."""
    logg = np.cumsum(np.log(alpha))
    decay = np.exp(np.minimum(logg[:, None] - logg[None, :], 0.0))
    lower = np.tril(beta[:, None] * decay * (k @ k.T), -1)
    rhs = beta[:, None] * (v - np.exp(logg)[:, None] * (k @ state))
    u = np.linalg.solve(np.eye(len(q)) + lower, rhs)
    out = np.exp(logg)[:, None] * (q @ state)
    out += np.tril(decay * (q @ k.T)) @ u
    end = np.exp(logg[-1]) * state + k.T @ (decay[-1, :, None] * u)
    return out, end


def conv_step(x, history, w):
    window = np.concatenate([history, x[None]], axis=0)
    raw = (window * w).sum(axis=0)
    return raw / (1 + np.exp(-raw)), window[1:].copy()


def rope(x, pos, theta=1e6):
    angles = pos * theta ** (-2 * np.arange(x.shape[-1] // 2) / x.shape[-1])
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * np.cos(angles) - b * np.sin(angles)
    out[..., 1::2] = a * np.sin(angles) + b * np.cos(angles)
    return out


def gqa_step(q, k, v, pos, cache):
    keys, values = cache
    keys = np.concatenate([keys, rope(k, pos)[None]], axis=0)
    values = np.concatenate([values, v[None]], axis=0)
    q = rope(q, pos)
    group = q.shape[0] // k.shape[0]
    idx = np.arange(q.shape[0]) // group
    score = np.einsum('hd,thd->ht', q, keys[:, idx]) / np.sqrt(q.shape[-1])
    p = np.exp(score - score.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    return np.einsum('ht,thd->hd', p, values[:, idx]), (keys, values)


class HybridSemantics(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.q = self.rng.normal(size=(9, 4))
        self.k = self.rng.normal(size=(9, 4))
        self.k /= np.linalg.norm(self.k, axis=-1, keepdims=True)
        self.v = self.rng.normal(size=(9, 3))
        self.a = self.rng.uniform(.2, .99, 9)
        self.b = self.rng.uniform(.1, .9, 9)
        self.s = self.rng.normal(size=(4, 3))

    def close(self, a, b):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)

    def run_part(self, fn, lo, hi, state):
        return fn(self.q[lo:hi], self.k[lo:hi], self.v[lo:hi],
                  self.a[lo:hi], self.b[lo:hi], state)

    def test_chunk_vs_recurrent_nonzero_state(self):
        y, s = self.run_part(recurrent, 0, 9, self.s)
        z, t = self.run_part(chunk, 0, 9, self.s)
        self.close(y, z)
        self.close(s, t)

    def test_irregular_chunks_and_prefix(self):
        y, s = self.run_part(recurrent, 0, 9, self.s)
        state = self.s
        out = []
        for lo, hi in ((0, 2), (2, 3), (3, 7), (7, 9)):
            z, state = self.run_part(chunk, lo, hi, state)
            out.append(z)
        self.close(y, np.concatenate(out))
        self.close(s, state)
        self.close(y[:4], self.run_part(chunk, 0, 4, self.s)[0])

    def test_gate_limits(self):
        q, k, v = self.q[:1], self.k[:1], self.v[:1]
        _, s = recurrent(q, k, v, [0.], [.7], self.s)
        self.close(s, .7 * np.outer(k[0], v[0]))
        _, s = recurrent(q, k, v, [.4], [0.], self.s)
        self.close(s, .4 * self.s)

    def test_matrix_rollback(self):
        _, prefix = self.run_part(recurrent, 0, 3, self.s)
        snapshot = prefix.copy()
        _, discarded = self.run_part(chunk, 3, 8, prefix)
        self.assertFalse(np.allclose(snapshot, discarded))
        y, s = self.run_part(recurrent, 3, 5, snapshot)
        reference, final = self.run_part(chunk, 0, 5, self.s)
        self.close(y, reference[3:])
        self.close(s, final)

    def test_conv_reset_and_rollback(self):
        x = self.rng.normal(size=(7, 5))
        w = self.rng.normal(size=(4, 5))
        zero = np.zeros((3, 5))
        history = zero.copy()
        for t in range(3):
            y, history = conv_step(x[t], history, w)
            window = np.concatenate([zero, x[:t+1]])[-4:]
            raw = (window * w).sum(axis=0)
            self.close(y, raw / (1 + np.exp(-raw)))
        saved = history.copy()
        _, scratch = conv_step(x[3] + 7, history, w)
        self.assertFalse(np.allclose(scratch, saved))
        y, h = conv_step(x[3], saved, w)
        raw = (x[:4] * w).sum(axis=0)
        self.close(y, raw / (1 + np.exp(-raw)))
        self.close(h, x[1:4])
        y, h = conv_step(x[4], zero, w)
        raw = x[4] * w[-1]
        self.close(y, raw / (1 + np.exp(-raw)))
        self.close(h, np.concatenate([zero[1:], x[4:5]]))

    def test_grouped_rope_attention_vs_explicit_causal(self):
        q = self.rng.normal(size=(6, 4, 4))
        k = self.rng.normal(size=(6, 2, 4))
        v = self.rng.normal(size=(6, 2, 4))
        cache = (np.empty((0, 2, 4)), np.empty((0, 2, 4)))
        for t in range(6):
            y, cache = gqa_step(q[t], k[t], v[t], t, cache)
            for h in range(4):
                scores = np.array([rope(q[t, h], t) @ rope(k[i, h//2], i)
                                   / 2 for i in range(t+1)])
                probs = np.exp(scores - scores.max())
                probs /= probs.sum()
                self.close(y[h], probs @ v[:t+1, h//2])
        self.assertEqual(cache[0].shape, (6, 2, 4))


if __name__ == '__main__':
    unittest.main(verbosity=2)
