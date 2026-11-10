# Keep pinned TVM unchanged and omit Node support from browser packages.
EMCC_LDFLAGS += -sENVIRONMENT=web,worker
