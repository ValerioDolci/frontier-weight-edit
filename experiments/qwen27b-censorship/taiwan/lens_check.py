import numpy as np
T = np.load("./taiwan/dec_TW_27b/attr_TW.npy"); NI = T.shape[1]; flat = T.ravel(); order = np.argsort(flat); rank = np.empty_like(order); rank[order] = np.arange(len(order))
for name, l, j in [("自古以来", 62, 2117), ("党的领导", 63, 6238), ("人民日报", 63, 5377), ("全面深化改革", 63, 3906), ("教育整顿", 63, 17330), ("扫黑", 63, 2678), ("中华民族伟大复兴", 63, 5947)]:
    f = l * NI + j; print("%-10s L%d n%d  attr_TW %+.3f  rango pro-copione %d su %d" % (name, l, j, flat[f], rank[f], len(flat)))
