# Session e50b — failed because of the harness, not counted as a replication

At pass 14 of 50 the editor's analysis kernel grew to 83 GB of RAM and the operating system's OOM killer stopped the whole
session (kernel and forward server). The editor then submitted its latest design, a layer-29 detector writing a flag that
a layer-47 neuron reads to lower the logits of the trapassato auxiliaries, without having been able to test it end to
end (it says so in `final_answer.md`). An earlier, single-stage version had been validated on practice data but was not
the submission.

Evaluated as submitted: test v2 DI 20 → 21 errors (0 fixed / 1 broken), test v3 DI unchanged (33 / 103).
Fix applied for later sessions: the kernel now caps its RAM (`WSP_KERNEL_MEM_GB`, default 56); over the cap an
allocation fails with an error and the session keeps running.
