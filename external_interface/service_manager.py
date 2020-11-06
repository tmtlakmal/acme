from external_interface.smarts_env import SMARTS_env
from external_interface.agent_handler import AgentHandler

class Manager():

    def __init__(self):
        self.env : SMARTS_env = SMARTS_env()
        self.env_handler : AgentHandler = AgentHandler(self.env)

    def first_step(self):
        self.env.init()

    def run(self, timesteps : int):
        self.first_step()
        for i in range(timesteps):
            self.env_handler.get_step_data()
            self.env_handler.step_agents()
            self.env.step()
            self.env_handler.fetch_agent_data()


    def online_run(self, timesteps : int):
        self.env_handler.add_common_env()

        self.first_step()
        self.env.step()
        result = self.env.get_result()

        for i in range(timesteps):
            for vehicle in result['vehicles']:
                if vehicle['externalControl']:
                    id = vehicle['vid']
                    self.env_handler.env_loops[0].set_id(id)
                    self.env_handler.env_loops[0].online_step()

            self.env.step()
            result = self.env.get_result()

    def close(self):
        self.env_handler.close()


if __name__ == '__main__':
    manager = Manager()
    #manager.run(30000)

    manager.online_run(4000)
    manager.close()
