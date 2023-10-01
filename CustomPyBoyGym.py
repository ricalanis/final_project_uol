from pyboy.pyboy import *
from AISettings.AISettingsInterface import AISettingsInterface
from gym import Env
from gym.spaces import Discrete, MultiDiscrete, Box
from pyboy.botsupport.constants import TILES
from pyboy.utils import WindowEvent
import numpy as np
import torch
import timm
model = timm.create_model("resnet50d", pretrained=True, features_only=True)

def rgb_to_grayscale(rgb_image):
    # Check if GPU is available, and move the RGB image to the GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rgb_tensor = torch.from_numpy(rgb_image).to(device)

    # Convert RGB to grayscale using luminance method
    grayscale_tensor = torch.sum(rgb_tensor * torch.tensor([0.299, 0.587, 0.114], device=device), dim=2, keepdim=True)

    # Convert the grayscale tensor back to a NumPy array
    grayscale_image = (grayscale_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

    return grayscale_image


class CustomPyBoyGym(PyBoyGymEnv):
    def __init__(self, pyboy, observation_type="tiles", action_type="toggle", simultaneous_actions=False, **kwargs):
        # Build pyboy game
        self.pyboy = pyboy
        if str(type(pyboy)) != "<class 'pyboy.pyboy.PyBoy'>":
            raise TypeError("pyboy must be a Pyboy object")

        # Build game_wrapper
        self.game_wrapper = pyboy.game_wrapper()
        if self.game_wrapper is None:
            raise ValueError(
                "You need to build a game_wrapper to use this function. Otherwise there is no way to build a reward function automaticaly."
            )
        self.last_fitness = self.game_wrapper.fitness

        # Building the action_space
        self._DO_NOTHING = WindowEvent.PASS
        self._buttons = [
            WindowEvent.PRESS_ARROW_UP, WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.PRESS_BUTTON_START
        ]
        self._button_is_pressed = {button: False for button in self._buttons}

        self._buttons_release = [
            WindowEvent.RELEASE_ARROW_UP, WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_START
        ]
        self._release_button = {button: r_button for button, r_button in zip(self._buttons, self._buttons_release)}

        self.actions = [self._DO_NOTHING] + self._buttons
        if action_type == "all":
            self.actions += self._buttons_release
        elif action_type not in ["press", "toggle"]:
            raise ValueError(f"action_type {action_type} is invalid")
        self.action_type = action_type

        if simultaneous_actions:
            raise NotImplementedError("Not implemented yet, raise an issue on GitHub if needed")
        else:
            self.action_space = Discrete(len(self.actions))

        # Building the observation_space
        if observation_type == "raw":
            screen = np.asarray(self.pyboy.botsupport_manager().screen().screen_ndarray())
            self.observation_space = MultiDiscrete(rgb_to_grayscale(screen))

        elif observation_type == "features":
            screen = np.asarray(self.pyboy.botsupport_manager().screen().screen_ndarray())
            screentensor = torch.as_tensor(np.array(screen, dtype=np.float32)).transpose(2,0)[None]
            output = model(screentensor)
            self.observation_space = MultiDiscrete(output[-4][0].transpose(0,2).sum(-1).detach().numpy().astype(int))

        elif observation_type in ["tiles", "compressed", "minimal"]:
            size_ids = TILES
            if observation_type == "compressed":
                try:
                    size_ids = np.max(self.game_wrapper.tiles_compressed) + 1
                except AttributeError:
                    raise AttributeError(
                        "You need to add the tiles_compressed attibute to the game_wrapper to use the compressed observation_type"
                    )
            elif observation_type == "minimal":
                try:
                    size_ids = np.max(self.game_wrapper.tiles_minimal) + 1
                except AttributeError:
                    raise AttributeError(
                        "You need to add the tiles_minimal attibute to the game_wrapper to use the minimal observation_type"
                    )
            nvec = size_ids * np.ones(self.game_wrapper.shape)
            self.observation_space = MultiDiscrete(nvec)
        else:
            raise NotImplementedError(f"observation_type {observation_type} is invalid")
        self.observation_type = observation_type

        self._started = False
        self._kwargs = kwargs 

    def _get_observation(self):
        if self.observation_type == "raw":
            screen = np.asarray(self.pyboy.botsupport_manager().screen().screen_ndarray())
            observation = rgb_to_grayscale(screen)
        elif self.observation_type == "features":
            screen = np.asarray(self.pyboy.botsupport_manager().screen().screen_ndarray())
            screentensor = torch.as_tensor(np.array(screen, dtype=np.float32)).transpose(2,0)[None]
            output = model(screentensor)
            observation = output[-4][0].transpose(0,2).sum(-1).detach().numpy().astype(int)
        elif self.observation_type in ["tiles", "compressed", "minimal"]:
            observation = self.game_wrapper._game_area_np(self.observation_type)
        else:
            raise NotImplementedError(f"observation_type {self.observation_type} is invalid")
        return observation   

    def step(self, list_actions):
        """
            Simultanious action implemention
        """
        info = {}

        previousGameState = self.aiSettings.GetGameState(self.pyboy)
        # (ricalanis) If we won versus boss, reset game
 
        if list_actions[0] == self._DO_NOTHING:
            pyboy_done = self.pyboy.tick()
        else:
            # release buttons if not pressed now but were pressed in the past
            for pressedFromBefore in [pressed for pressed in self._button_is_pressed if self._button_is_pressed[pressed] == True]: # get all buttons currently pressed
                if pressedFromBefore not in list_actions:
                    release = self._release_button[pressedFromBefore]
                    self.pyboy.send_input(release)
                    self._button_is_pressed[release] = False

            # press buttons we want to press
            for buttonToPress in list_actions:
                self.pyboy.send_input(buttonToPress)
                self._button_is_pressed[buttonToPress] = True # update status of the button

            pyboy_done = self.pyboy.tick()

        # reward 
        reward = self.aiSettings.GetReward(previousGameState, self.pyboy)

        observation = self._get_observation()

        boss_defeated = self.pyboy.get_memory_value(0xD093)==0 #(ricalanis) used 3830 score but did not hold as useful

        done = pyboy_done or self.pyboy.game_wrapper().game_over() or boss_defeated

        if boss_defeated: print("*******Won vs Boss***********")

        return observation, reward, done, info

    def setAISettings(self, aisettings: AISettingsInterface):
        self.aiSettings = aisettings

    def reset(self):
        """ Reset (or start) the gym environment thrUought the game_wrapper """
        if not self._started:
            self.game_wrapper.start_game(**self._kwargs)
            self._started = True
        else:
            self.game_wrapper.reset_game()
        self.pyboy.load_state(open("games/KIRBY.gb.state", "rb"))

        # release buttons if not pressed now but were pressed in the past
        for pressedFromBefore in [pressed for pressed in self._button_is_pressed if self._button_is_pressed[pressed] == True]: # get all buttons currently pressed
            self.pyboy.send_input(self._release_button[pressedFromBefore])
        self.button_is_pressed = {button: False for button in self._buttons} # reset all buttons

        return self._get_observation()