import copy
import hashlib
import json
from pathlib import Path
import types
import unittest

import g_outgoing_driver_query as bridge


class DriverBridgeTests(unittest.TestCase):
    def fixture(self, enabled=True):
        rows = [[2, 0, 1, 1], [0, 1, 901], [0, 2, 101], [0, 3, 201], [0, 4, 902],
                [1, 1, 2, 0, 0, 0], [1, 2, 3, 1 if enabled else 0, 0, 1 if enabled else 0], [1, 3, 4, 0, 0, 0]]
        return rows, dict(namespace=bridge.IDENTITY_NAMESPACE, values=[101, 102]), dict(namespace=bridge.IDENTITY_NAMESPACE, values=[201])

    def test_mixed_foreign_nodes_and_both_modes(self):
        for enabled in (False, True):
            rows, primary, consumer = self.fixture(enabled)
            proof = bridge.validate_edges(rows, primary, consumer, enabled)
            self.assertEqual(proof["producer"], [2, 101])
            self.assertEqual(proof["neighbor_kernels"], [[1, 901], [4, 902]])
            self.assertTrue(proof["edge_data_nonnull"])

    def test_identity_namespace_and_old_stub_negatives(self):
        rows, primary, consumer = self.fixture()
        for namespace in ("host_stub", "CUkernel", "cudaKernel_t"):
            with self.assertRaisesRegex(AssertionError, "NAMESPACE"):
                bridge.validate_edges(rows, dict(primary, namespace=namespace), consumer, True)
        with self.assertRaisesRegex(AssertionError, "OS_NODE_IDENTITY"):
            bridge.validate_edges(rows, dict(primary, values=[1001, 1002]), consumer, True)

    def test_existing_edge_gates_not_weakened(self):
        rows, primary, consumer = self.fixture()
        for index, replacement in ((5, [1, 1, 2, 1, 0, 1]), (6, [1, 2, 3, 0, 0, 0]), (7, [1, 3, 4, 1, 0, 1])):
            changed = copy.deepcopy(rows)
            changed[index] = replacement
            with self.assertRaises(AssertionError):
                bridge.validate_edges(changed, primary, consumer, True)

    def call(self, query, messages):
        return bridge.inspect_capture(types.SimpleNamespace(raw_cuda_graph=lambda: 123),
                                      types.SimpleNamespace(seam_driver_primary_identity=lambda row, arm: [101, 102]), query,
                                      row=12, arm=3, enabled=True, resources={"tiny": [145, 0]},
                                      build_census=[{"name": "mock", "flags": ["sm_90a"]}],
                                      emit=lambda text, **kwargs: messages.append(text))

    def test_prefix_survives_foreign_node_failure(self):
        messages = []
        def failed(address, row, arm):
            raise RuntimeError(f"api=cuGraphKernelNodeGetParams_v2 code=98 node=4 row={row} arm={arm}")
        query = types.SimpleNamespace(seam_driver_unpool_identity=lambda row, arm: 201, seam_driver_graph_edges=failed)
        with self.assertRaisesRegex(RuntimeError, "node=4 row=12 arm=3"):
            self.call(query, messages)
        prefix = json.loads(messages[0].split(" ", 1)[1])
        self.assertEqual(prefix["resources"]["tiny"], [145, 0])
        self.assertEqual(prefix["build_census"][0]["flags"], ["sm_90a"])
        self.assertIn("node=4", messages[1])
        self.assertIn("seam_driver_graph_edges", messages[1])

    def test_real_scratch_entry_calls_only_canonical_exports(self):
        rows, _, _ = self.fixture()
        calls = []
        query = types.SimpleNamespace(seam_driver_unpool_identity=lambda row, arm: calls.append(("identity", row, arm)) or 201,
                                     seam_driver_graph_edges=lambda address, row, arm: calls.append(("graph", address, row, arm)) or rows)
        result = self.call(query, [])
        self.assertEqual(calls, [("identity", 12, 3), ("graph", 123, 12, 3)])
        self.assertEqual(result["rows"], rows)

    def test_source_negatives_and_failed_pins(self):
        self.assertEqual(bridge.QUERY_EXTRA_LDFLAGS, ("-lcuda",))
        for source in (bridge.PRIMARY, bridge.SECONDARY):
            self.assertIn("cudaGetFuncBySymbol(&function,symbol)", source)
            self.assertIn("reinterpret_cast<CUfunction>(function)", source)
            self.assertNotIn("__global__", source)
        self.assertNotIn("cudaGraphKernelNodeGetParams(", bridge.SECONDARY)
        self.assertIn("cuGraphKernelNodeGetParams(node,&params)", bridge.SECONDARY)
        self.assertIn("for(auto node:nodes)", bridge.SECONDARY)
        self.assertNotIn("continue;", bridge.SECONDARY)
        self.assertIn("cuGraphGetEdges(graph,from.data(),to.data(),edges.data(),&count)", bridge.SECONDARY)
        self.assertIn("non_CUfunction_representation", bridge.SECONDARY)
        root = Path(__file__).resolve().parents[1]
        for path, digest in ((".build/outgoing-pair/check.py", "f25d714e928835fba07741ca56fa8783137032f268652c22ec4f5511d771bc9a"),
                             ("tools/g_outgoing_pair_read.py", "5435108e9eb729c7fe1f705244845cc7c916b9af780081e682c83ae81f736c1b")):
            self.assertEqual(hashlib.sha256((root / path).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
