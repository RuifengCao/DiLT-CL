"""Parameter- and memory-cost accounting for the continual-learning methods.

The useful comparison for a parameter-isolation method like DiLT-CL is accuracy
relative to an independent-task reference at a stated storage cost. This
module computes,
for each method, the storage required to serve all ``T`` tasks after training:

* real-valued parameters (``float32``),
* binary-mask bits (``1`` bit per masked weight per task),
* optional continual-training state (for ER / DER++ / EWC).

Everything here is a pure function of scalars (the backbone parameter count
``P``, the number of tasks ``T``, mask densities, buffer size, ...), so it can
be unit-tested without PyTorch and produces the same numbers whether or not a
GPU run is available.

Accounting conventions
----------------------
* ``float32`` => 4 bytes; one mask entry => 1 bit (a dense bitmap per task).
* The default ``state_scope="inference"`` counts only state needed to serve
  predictions. EWC statistics and replay buffers are then excluded, just like
  LwF's teacher. ``state_scope="resume"`` additionally counts the actual
  implementation's per-task EWC Fishers/anchors and replay buffers.
* **LwF** keeps only the backbone at deployment (its teacher snapshot is a
  train-time transient).
* **DiLT-CL** uses the core alone for task 0 and stores ``T-1`` experts, hence
  one core plus ``T-1`` sparse expert tensors and ``T`` masks. The current code
  has no compact exporter and keeps dense tensors at runtime: this module is an
  *analytic compact-format estimate*, not a measured checkpoint size.
* **PackNet** stores one packed backbone plus ``T`` weight-mask bitmaps.
* **SupSup** stores only ``T`` mask bitmaps: its backbone is a fixed random net
  that is regenerable from a single integer seed, so we count it as free
  (set ``store_backbone=True`` to count it explicitly).
"""

from dataclasses import dataclass

FLOAT_BYTES = 4           # float32
_CANON_ORDER = ["finetune", "ewc", "lwf", "er", "derpp",
                "packnet", "supsup", "wsn", "diltcl", "independent"]


@dataclass
class MethodCost:
    """Storage cost of one method, in parameters / mask bits / buffer floats."""

    name: str
    float_params: float
    mask_bits: float = 0.0
    buffer_floats: float = 0.0
    note: str = ""

    def bytes(self):
        """Total deployment storage in bytes."""
        return (self.float_params * FLOAT_BYTES
                + self.mask_bits / 8.0
                + self.buffer_floats * FLOAT_BYTES)

    def megabytes(self):
        """Total deployment storage in mebibytes (MiB)."""
        return self.bytes() / (1024.0 ** 2)


# ---------------------------------------------------------------------------
# Per-method cost builders
# ---------------------------------------------------------------------------

def single_backbone(name, params, note=""):
    return MethodCost(name, float_params=params, note=note)


def ewc_cost(params, num_tasks=1, state_scope="inference"):
    if state_scope == "inference":
        return MethodCost("EWC", float_params=params,
                          note="inference only; Fishers/anchors discarded")
    if state_scope == "resume":
        return MethodCost(
            "EWC", float_params=(1 + 2 * num_tasks) * params,
            note=f"resume state: backbone + {num_tasks} Fishers + anchors")
    raise ValueError("state_scope must be 'inference' or 'resume'")


def lwf_cost(params):
    return MethodCost("LwF", float_params=params,
                      note="teacher snapshot is train-time only")


def replay_cost(name, params, buffer_size, sample_numel, num_classes=0,
                state_scope="inference"):
    if state_scope not in {"inference", "resume"}:
        raise ValueError("state_scope must be 'inference' or 'resume'")
    buf = (buffer_size * sample_numel + buffer_size * num_classes
           if state_scope == "resume" else 0)
    note = (f"resume buffer={buffer_size} samples"
            if state_scope == "resume" else "inference only; buffer discarded")
    if num_classes and state_scope == "resume":
        note += " + stored logits"
    return MethodCost(name, float_params=params, buffer_floats=buf, note=note)


def independent_cost(params, num_tasks):
    return MethodCost("Independent", float_params=num_tasks * params,
                      note=f"{num_tasks} full models")


def packnet_cost(params, num_tasks):
    return MethodCost("PackNet", float_params=params,
                      mask_bits=num_tasks * params,
                      note="packed backbone + per-task weight masks")


def wsn_cost(params, num_tasks):
    return MethodCost("WSN", float_params=params,
                      mask_bits=num_tasks * params,
                      note="trained backbone + per-task weight masks")


def supsup_cost(params, num_tasks, store_backbone=False):
    note = ("fixed random backbone (regenerable from a seed)"
            if not store_backbone else "fixed backbone stored explicitly")
    return MethodCost("SupSup",
                      float_params=(params if store_backbone else 0.0),
                      mask_bits=num_tasks * params, note=note)


def diltcl_cost(params, num_tasks, core_density, expert_density):
    """Analytic compact-format storage for DiLT-CL.

    Task 0 is served by the shared core alone, so only ``num_tasks - 1`` experts
    exist. (The v2.0 formula counted ``num_tasks`` experts and therefore
    over-reported the cost.) Masks are counted as one bit per parameter, which
    the dense PyTorch checkpoint does not achieve; use
    :mod:`diltcl.compact_export` to produce -- and logit-verify -- an artifact
    that actually realises this estimate.

    Args:
        params: Trainable parameters in the backbone.
        num_tasks: Number of tasks in the sequence.
        core_density: Fraction of parameters kept by the core mask.
        expert_density: Fraction kept by each expert mask.
    """
    num_experts = max(0, num_tasks - 1)
    floats = (core_density + num_experts * expert_density) * params
    return MethodCost(
        "DiLT-CL (ours)", float_params=floats,
        mask_bits=num_tasks * params,
        note=(f"analytic compact estimate: core (d={core_density:.2f}) + "
              f"{num_experts} experts (d={expert_density:.2f})"))


def build_default_suite(methods, params, num_tasks, *, buffer_size,
                        sample_numel, num_classes, core_density,
                        expert_density, supsup_store_backbone=False,
                        state_scope="inference"):
    """Build :class:`MethodCost` entries for ``methods`` in canonical order.

    Args:
        methods: Iterable of method names to include.
        params: Backbone trainable-parameter count ``P`` (incl. biases).
        num_tasks: Number of tasks ``T``.
        buffer_size: Replay-buffer capacity (samples).
        sample_numel: Number of scalars per stored input sample.
        num_classes: Classes per task (DER++ stores this many logits/sample).
        core_density: Measured DiLT-CL core density in ``[0, 1]``.
        expert_density: Measured DiLT-CL per-task expert density in ``[0, 1]``.
        supsup_store_backbone: Count SupSup's fixed backbone if ``True``.

    Returns:
        List of :class:`MethodCost`, ordered canonically.
    """
    present = set(methods)
    builders = {
        "finetune": lambda: single_backbone("Finetune", params,
                                            "lower bound"),
        "ewc": lambda: ewc_cost(params, num_tasks, state_scope),
        "lwf": lambda: lwf_cost(params),
        "er": lambda: replay_cost("ER", params, buffer_size, sample_numel, 0,
                                    state_scope),
        "derpp": lambda: replay_cost("DER++", params, buffer_size,
                                     sample_numel, num_classes, state_scope),
        "packnet": lambda: packnet_cost(params, num_tasks),
        "supsup": lambda: supsup_cost(params, num_tasks,
                                      store_backbone=supsup_store_backbone),
        "wsn": lambda: wsn_cost(params, num_tasks),
        "diltcl": lambda: diltcl_cost(params, num_tasks, core_density,
                                      expert_density),
        "independent": lambda: independent_cost(params, num_tasks),
    }
    return [builders[m]() for m in _CANON_ORDER if m in present]


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _reference_bytes(costs, reference="Independent"):
    for c in costs:
        if c.name == reference:
            return c.bytes()
    return None


def format_cost_table_text(costs, params, reference="Independent"):
    """Render a human-readable cost table.

    Columns: method, real params as a multiple of ``P``, mask size (Mbit),
    buffer (MiB), total (MiB), and (if present) the ratio to the reference.
    """
    ref = _reference_bytes(costs, reference)
    lines = []
    header = (f"{'Method':<18}{'Params(xP)':>12}{'Masks(Mbit)':>13}"
              f"{'Buffer(MiB)':>13}{'Total(MiB)':>12}")
    if ref:
        header += f"{'xReference':>11}"
    lines.append(header)
    lines.append("-" * len(header))
    for c in costs:
        row = (f"{c.name:<18}{c.float_params / params:>12.2f}"
               f"{c.mask_bits / 1e6:>13.2f}"
               f"{c.buffer_floats * FLOAT_BYTES / (1024 ** 2):>13.2f}"
               f"{c.megabytes():>12.2f}")
        if ref:
            row += f"{c.bytes() / ref:>11.3f}"
        lines.append(row)
    lines.append("")
    lines.append(f"Backbone P = {params:,} trainable parameters "
                 f"({params * FLOAT_BYTES / (1024 ** 2):.2f} MiB as float32).")
    return "\n".join(lines)


def format_cost_table_latex(costs, params, num_tasks, benchmark="",
                            reference="Independent", state_scope="inference"):
    """Render the cost table as booktabs LaTeX."""
    ref = _reference_bytes(costs, reference)
    display = {"cifar": "Split-CIFAR-100",
               "pmnist": "Permuted-MNIST"}.get(benchmark, benchmark)
    scope_text = ("inference-serving state" if state_scope == "inference"
                  else "resume-learning state")
    cap = (f"Analytic {scope_text} estimate to serve all tasks"
           + (f" on {display}" if display else "")
           + f" (backbone $P={params:,}$ parameters, $T={num_tasks}$ tasks). "
           "``Params'' is real-valued storage as a multiple of one backbone; "
           "``Masks'' is binary supermask storage; ``Total'' assumes "
           "\\texttt{float32} weights, 1-bit masks, and ideal sparse encoding "
           "for DiLT-CL (not yet implemented as an exporter).")
    rows = []
    for c in costs:
        cells = [c.name,
                 f"{c.float_params / params:.2f}",
                 f"{c.mask_bits / 1e6:.2f}",
                 f"{c.buffer_floats * FLOAT_BYTES / (1024 ** 2):.2f}",
                 f"{c.megabytes():.2f}"]
        if ref:
            cells.append(f"{c.bytes() / ref:.3f}")
        rows.append(" & ".join(cells) + r" \\")
    colspec = "l ccc c" + (" c" if ref else "")
    head = ["Method", r"Params ($\times P$)", "Masks (Mbit)",
            "Buffer (MiB)", "Total (MiB)"]
    if ref:
        head.append(r"$\times$ Ref.")
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n"
        f"\\label{{tab:memory{('_' + benchmark) if benchmark else ''}}}\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
        + " & ".join(head) + r" \\" + "\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
