from external_interface.smarts_env import SMARTS_env
from external_interface.agent_handler import AgentHandler
import time
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

    def online_run(self, timesteps : int, controller="RL"):
        self.env_handler.add_common_env(controller)

        self.first_step()
        self.env.step()
        result = self.env.get_result()

        for i in range(timesteps):
            index = 0
            for vehicle in result['vehicles']:
                if vehicle['externalControl']:
                    index += 1
                    id = vehicle['vid']
                    self.env_handler.env_loops[0].set_id(id)
                    self.env_handler.env_loops[0].online_step()
            self.env.step()
            result = self.env.get_result()

    def mixed_run(self, timesteps):
        self.env_handler.create_loop(1, trained=True)
        self.env_handler.create_loop(2, trained=False)

        self.env_handler.env_loops[0].load()
        #self.env_handler.env_loops[1].load()

        self.first_step()
        self.env.step()
        result = self.env.get_result()

        for i in range(timesteps):
            for vehicle in result['vehicles']:
                if vehicle['externalControl']:
                    if vehicle['vid'] == 1:
                        self.env_handler.env_loops[0].online_step()
                    if vehicle['vid'] == 2:
                        self.env_handler.env_loops[1].run_step()

            self.env.step()

            for vehicle in result['vehicles']:
                if vehicle['externalControl']:
                    if vehicle['vid'] == 2:
                        self.env_handler.env_loops[1].fetch_data()

            result = self.env.get_result()

    def mixed_gurobi_run(self, timesteps):
        self.env_handler.create_loop(1, trained=False, gurobi=True)
        self.env_handler.create_loop(2, trained=False, gurobi=True)

        #self.env_handler.env_loops[0].load()
        #self.env_handler.env_loops[1].load()

        self.first_step()
        self.env.step()
        result = self.env.get_result()

        for i in range(timesteps):
            for vehicle in result['vehicles']:
                if vehicle['externalControl']:
                    if vehicle['vid'] == 1:
                        self.env_handler.env_loops[0].run_step()
                    if vehicle['vid'] == 2:
                        self.env_handler.env_loops[1].run_step()

            self.env.step()

            for vehicle in result['vehicles']:
                if vehicle['externalControl']:
                    if vehicle['vid'] == 1:
                        self.env_handler.env_loops[0].fetch_data()
                    if vehicle['vid'] == 2:
                        self.env_handler.env_loops[1].fetch_data()

            result = self.env.get_result()



    def close(self):
        self.env_handler.close()


if __name__ == '__main__':
    manager = Manager()
    manager.online_run(30000, controller="Gurobi")
    manager.close()
