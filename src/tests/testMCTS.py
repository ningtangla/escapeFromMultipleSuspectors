import sys
sys.path.append('..')
import unittest
import numpy as np
from ddt import ddt, data, unpack
from anytree import AnyNode as Node

# Local import
from algorithms.stochasticMCTS import CalculateScore, SelectChild, Expand, RollOut, backup, GetActionPrior, InitializeChildren
from simple1DEnv import TransitionFunction, RewardFunction, Terminal


@ddt
class TestMCTS(unittest.TestCase):
    def setUp(self):
        # Env param
        bound_low = 0
        bound_high = 7
        self.transition = TransitionFunction(bound_low, bound_high)

        self.action_space = [-1, 1]
        self.num_action_space = len(self.action_space)
        self.action_prior_func = GetActionPrior(self.action_space)

        step_penalty = -1
        catch_reward = 1
        self.target_state = bound_high
        self.isTerminal = Terminal(self.target_state)

        self.c_init = 0
        self.c_base = 1
        self.calculateScore = CalculateScore(self.c_init, self.c_base)

        self.selectChild = SelectChild(self.calculateScore)
        
        init_state = 3
        level1_0_state = self.transition(init_state, action=0)
        level1_1_state = self.transition(init_state, action=1)
        self.default_action_prior = 0.5

        self.root = Node(id={1: init_state}, num_visited=1, sum_value=0, action_prior=self.default_action_prior, is_expanded=True)
        self.level1_0 = Node(parent=self.root, id={0: level1_0_state}, num_visited=2, sum_value=5, action_prior=self.default_action_prior, is_expanded=False)
        self.level1_1 = Node(parent=self.root, id={1: level1_1_state}, num_visited=3, sum_value=10, action_prior=self.default_action_prior, is_expanded=False)
        
        self.initializeChildren = InitializeChildren(self.action_space, self.transition, self.action_prior_func)
        self.expand = Expand(self.isTerminal, self.initializeChildren)

    @data((0, 1, 0, 1, 0), (1, 1, 0, 1, np.log(3)/2), (1, 1, 1, 1, 1 + np.log(3)/2))
    @unpack
    def testCalculateScore(self, parent_visit_number, self_visit_number, sum_value, action_prior, groundtruth_score):
        curr_node = Node(num_visited = parent_visit_number)
        child = Node(num_visited = self_visit_number, sum_value = sum_value, action_prior = action_prior)
        score = self.calculateScore(curr_node, child)
        self.assertEqual(score, groundtruth_score)

    @data((1, 1, 1, 1, 100))
    @unpack
    def testVisitValueEffectsOnSelectChild(self, firstChildVisited, firstChildSumValue, secondChildVisited, secondChildSumValue, maxSelectTimes):
        curr_node = Node(num_visited = 1)
        first_child = Node(parent = curr_node, id = 'first', num_visited = firstChildVisited, sum_value = firstChildSumValue, action_prior = 0.5, is_expanded = False)
        second_child = Node(parent = curr_node, id = 'second', num_visited = secondChildVisited, sum_value = secondChildSumValue, action_prior = 0.5, is_expanded = False)
        old_child_id = 'none'
        childIndexChangeFrequency = 2
        for selectIndex in range(maxSelectTimes):
            new_child = self.selectChild(curr_node)
            new_child.sum_value -= 1
            if selectIndex % childIndexChangeFrequency == 1:
                self.assertTrue(new_child.id != old_child_id)
            old_child_id = new_child.id

    @data((1, 1, 1, 1, 100))
    @unpack
    def testVisitNumEffectsOnSelectChild(self, firstChildVisited, firstChildSumValue, secondChildVisited, secondChildSumValue, maxSelectTimes):
        curr_node = Node(num_visited = 1)
        
        first_child = Node(parent = curr_node, id = 'first', num_visited = firstChildVisited, sum_value = firstChildSumValue, action_prior = 0.5, is_expanded = False)
        second_child = Node(parent = curr_node, id = 'second', num_visited = secondChildVisited, sum_value = secondChildSumValue, action_prior = 0.5, is_expanded = False)
        old_child_id = 'none'
        childIndexChangeFrequency = 2
        for selectIndex in range(maxSelectTimes):
            new_child = self.selectChild(curr_node)
            new_child.num_visited += 1
            if selectIndex % childIndexChangeFrequency == 1:
                self.assertTrue(new_child.id != old_child_id)
            old_child_id = new_child.id
     
    @data((3, True, [2, 4]), (0, True, [0, 1]), (7, True, [6, 7]))
    @unpack
    def testExpand(self, state, has_children, child_states):

        leaf_node = Node(id = {1: state}, num_visited = 1, sum_value = 1, action_prior = 0.5, is_expanded = False)
        expanded_node = self.expand(leaf_node)

        calc_has_children = (len(expanded_node.children) != 0)
        self.assertEqual(has_children, calc_has_children)

        for child_index, child in enumerate(expanded_node.children):
            cal_child_state = list(child.id.values())[0]
            gt_child_state = child_states[child_index]
            self.assertEqual(gt_child_state, cal_child_state)

    @data((4, 3, 0.125), (3, 4, 0.25))
    @unpack
    def testRollout(self, max_rollout_step, init_state, gt_sum_value):
        max_iteration = 1000

        target_state = 6
        isTerminal = Terminal(target_state)

        catch_reward = 1
        step_penalty = 0
        reward_func = RewardFunction(step_penalty, catch_reward, isTerminal)

        rollout_policy = lambda state: np.random.choice(self.action_space)
        leaf_node = Node(id={1: init_state}, num_visited=1, sum_value=0, action_prior=self.default_action_prior, is_expanded=True)
        rollout = RollOut(rollout_policy, max_rollout_step, self.transition, reward_func, isTerminal)
        stored_reward = []
        for curr_iter in range(max_iteration):
            stored_reward.append(rollout(leaf_node))
        
        calc_sum_value = np.mean(stored_reward)

        self.assertAlmostEqual(gt_sum_value, calc_sum_value, places=1)

    @data((5, [3,4], [2,1], [8,9], [3,2]))
    @unpack
    def testBackup(self, value, prev_sum_values, prev_visit_nums, new_sum_values, new_visit_nums):
        node_list = []
        for prev_sum_value, prev_visit_num in zip(prev_sum_values, prev_visit_nums):
            node_list.append(Node(id = {1: 4}, num_visited = prev_visit_num, sum_value = prev_sum_value, action_prior = 0.5, is_expanded = False))

        backup(value, node_list)
        cal_sum_values = [node.sum_value for node in node_list]
        cal_visit_nums = [node.num_visited for node in node_list]

        self.assertTrue(np.all(cal_sum_values == new_sum_values))
        self.assertTrue(np.all(cal_visit_nums == new_visit_nums))


if __name__ == "__main__":
    unittest.main()


