import unittest
from unittest.mock import patch, MagicMock
from src.main import create_supervisor_agent, State, SupervisorOutput
from src.models.openai import ModelTier

# filepath: /home/prodev/devtime/dynarch/src/test_main.py

class TestCreateSupervisorAgent(unittest.TestCase):
    @patch("src.main.create_supervisor")
    @patch("src.main.get_openai_model")
    def test_create_supervisor_agent_root(self, mock_get_openai_model, mock_create_supervisor):
        # Mocking dependencies
        mock_model = MagicMock()
        mock_get_openai_model.return_value = mock_model
        mock_agent_flow = MagicMock()
        mock_create_supervisor.return_value = mock_agent_flow
        mock_agent_flow.compile.return_value.invoke.return_value = MagicMock(
            messages=["Test message"], tasks=[]
        )

        # Test data
        state = State()
        config = MagicMock()
        supervisor_name = ""
        agent_name = "root"
        prompt = "Test prompt"

        # Call the function
        result = create_supervisor_agent(state, config, supervisor_name, agent_name, prompt)

        # Assertions
        mock_create_supervisor.assert_called_once_with(
            agents=[],
            model=mock_model,
            prompt=state.tools[0].__doc__,
            response_format=SupervisorOutput,
            tools=state.tools,
            supervisor_name=agent_name,
        )
        self.assertEqual(result.messages, ["Test message"])
        self.assertEqual(result.tasks, [])

    @patch("src.main.create_supervisor")
    @patch("src.main.get_openai_model")
    def test_create_supervisor_agent_non_root(self, mock_get_openai_model, mock_create_supervisor):
        # Mocking dependencies
        mock_model = MagicMock()
        mock_get_openai_model.return_value = mock_model
        mock_agent_flow = MagicMock()
        mock_create_supervisor.return_value = mock_agent_flow
        mock_agent_flow.compile.return_value.invoke.return_value = MagicMock(
            messages=["Another message"], tasks=["Task 1"]
        )

        # Test data
        state = State()
        state.agents.add_child(state.agents)  # Add a dummy child agent
        config = MagicMock()
        supervisor_name = "parent"
        agent_name = "child"
        prompt = "Another test prompt"

        # Call the function
        result = create_supervisor_agent(state, config, supervisor_name, agent_name, prompt)

        # Assertions
        mock_create_supervisor.assert_called_once()
        self.assertEqual(result.messages, ["Another message"])
        self.assertEqual(result.tasks, ["Task 1"])

if __name__ == "__main__":
    unittest.main()