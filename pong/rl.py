import pygame
import sys
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 400
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FPS = 60

# Paddle constants
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 60
PADDLE_SPEED = 5

# Ball constants
BALL_SIZE = 10
BALL_SPEED_X = 5
BALL_SPEED_Y = 3

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1

# Create the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong Game")

# Define Paddle class
class Paddle(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((PADDLE_WIDTH, PADDLE_HEIGHT))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

    def update(self):
        pass  # Paddle movement is handled by the RL agent

# Define Ball class
class Ball(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((BALL_SIZE, BALL_SIZE))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.speed_x = BALL_SPEED_X * random.choice([-1, 1])
        self.speed_y = BALL_SPEED_Y * random.choice([-1, 1])

    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        # Bounce off the top and bottom
        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.speed_y = -self.speed_y

        # Reset ball position if it goes out of bounds
        if self.rect.left <= 0 or self.rect.right >= WIDTH:
            self.rect.center = (WIDTH // 2, HEIGHT // 2)
            self.speed_x *= random.choice([-1, 1])

# Create sprites
all_sprites = pygame.sprite.Group()
paddle1 = Paddle(20, HEIGHT // 2)
paddle2 = Paddle(WIDTH - 20, HEIGHT // 2)
ball = Ball()
all_sprites.add(paddle1, paddle2, ball)

# Initialize Q-table dynamically based on state space
state = (paddle1.rect.y, ball.rect.y)  # Initial state to determine the state space size
state_space_size = len(state)
action_space_size = 2  # Adjust based on the number of possible actions
q_table = np.zeros((state_space_size, action_space_size))

# Convert state to a discrete value (for simplicity in this example)
def get_discrete_state(state):
    return tuple(np.round(state, decimals=0).astype(int))

# Convert action to a discrete value (for simplicity in this example)
def get_discrete_action(action):
    return int(round(action))

# Map actions to paddle movements
def take_action(paddle, action):
    if action == 0:
        paddle.rect.y -= PADDLE_SPEED
    elif action == 1:
        paddle.rect.y += PADDLE_SPEED

# Calculate reward based on the game state
def calculate_reward(ball, paddle):
    if pygame.sprite.collide_rect(ball, paddle):
        return 1  # Positive reward for hitting the ball
    else:
        return 0

# Q-learning algorithm
def q_learning(state, action, reward, next_state):
    current_q = q_table[state][action]
    best_future_q = np.max(q_table[next_state])
    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * best_future_q)
    q_table[state][action] = new_q

# Game loop
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # RL agent actions
    state = get_discrete_state([paddle1.rect.y, ball.rect.y])
    action = np.argmax(q_table[state])

    # Apply action to the RL agent
    take_action(paddle1, action)

    # Update game elements
    all_sprites.update()

    # Ball and paddle collisions
    if pygame.sprite.collide_rect(ball, paddle2):
        # RL agent controlled paddle
        ball.speed_x = -ball.speed_x

        # Calculate reward and update Q-table
        reward = calculate_reward(ball, paddle1)
        next_state = get_discrete_state([paddle1.rect.y, ball.rect.y])
        q_learning(state, action, reward, next_state)

    # Draw background
    screen.fill(BLACK)

    # Draw sprites
    all_sprites.draw(screen)

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    clock.tick(FPS)
