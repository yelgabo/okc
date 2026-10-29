import re
import sys

insts = ["1,1", "1,3", "1,11", "1,13", "1,17", "1,49", "8,1", "8,3", "8,11", "8,15", "8,19", "8,51", "8,115"]
text = open(sys.argv[1]).read()
text = "#define TORCH_CHECK(cond, ...) (void)(cond)\n" + re.sub(r'#include <(torch|ATen)/[^>]*>\n', '', text)
head, sep, tail = text.rpartition("}  // namespace")
inst = "".join(f"template struct Instance<{x}>;\n" for x in insts)
open(sys.argv[2], "w").write(head + inst + sep + "\n")
