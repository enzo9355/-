import ast
import inspect
import unittest


class TWCalculatedColumnContractTests(unittest.TestCase):
    def test_calculated_columns_match_calc_all_assignments_in_order(self):
        from stock_papi.quant import features

        tree = ast.parse(inspect.getsource(features.calc_all))
        assigned = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "frame"
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    assigned.append((node.lineno, target.slice.value))
        names = tuple(name for _, name in sorted(assigned))
        self.assertEqual(features.CALCULATED_COLUMNS, names)
        self.assertEqual(len(names), 20)
