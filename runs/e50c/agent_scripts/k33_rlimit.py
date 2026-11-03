import resource
print("RLIMIT_AS", resource.getrlimit(resource.RLIMIT_AS))
print("RLIMIT_DATA", resource.getrlimit(resource.RLIMIT_DATA))
print("RLIMIT_RSS", resource.getrlimit(resource.RLIMIT_RSS))
