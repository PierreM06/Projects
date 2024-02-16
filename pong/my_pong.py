import numpy as np
import pygame
import sys
import math
import random

def draw_line_dashed(surface, color, start_pos, end_pos, width = 1, dash_length = 10, exclude_corners = True):

    # convert tuples to numpy arrays
    start_pos = np.array(start_pos)
    end_pos   = np.array(end_pos)

    # get euclidian distance between start_pos and end_pos
    length = np.linalg.norm(end_pos - start_pos)

    # get amount of pieces that line will be split up in (half of it are amount of dashes)
    dash_amount = int(length / dash_length)

    # x-y-value-pairs of where dashes start (and on next, will end)
    dash_knots = np.array([np.linspace(start_pos[i], end_pos[i], dash_amount) for i in range(2)]).transpose()

    return [pygame.draw.line(surface, color, tuple(dash_knots[n]), tuple(dash_knots[n+1]), width)
            for n in range(int(exclude_corners), dash_amount - int(exclude_corners), 2)]

# Initialize Pygame
pygame.init()

# Constants
width, height = 600, 400
white = (255, 255, 255)
black = (0,0,0)
fps = 60

paddle_speed = 3
paddle1 = pygame.Rect(40-12//2, height//2-80//2, 12, 80)
paddle2 = pygame.Rect(width-40-12//2, height//2-80//2, 12, 80)

paddle1_score = 0
paddle2_score = 0

# Set up the ball
ball_radius = 10
ball_speed = 5
ball_direction = random.randint(0,1)
if ball_direction == 0:
    ball_angle = math.radians(random.uniform(-45, 45))  # Random angle between -45 and 45 degrees
elif ball_direction == 1:
    ball_angle = math.radians(random.uniform(180-45, 180+45))  # Random angle between -45 and 45 degrees
ball_speed_x = ball_speed * math.cos(ball_angle)
ball_speed_y = ball_speed * math.sin(ball_angle)
ball_x, ball_y = width // 2, height // 2

# Track the ball's speed
current_ball_speed = ball_speed

# Set up speed increment
speed_increment = 0.05

# Create the game window
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Pong Game")

# Game loop
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Get the state of all keys
    keys = pygame.key.get_pressed()

    # Move paddle1 (left paddle)
    if keys[pygame.K_w]:
        paddle1.move_ip(0, -paddle_speed)
    if keys[pygame.K_s]:
        paddle1.move_ip(0, paddle_speed)

    # Move paddle2 (right paddle)
    if keys[pygame.K_UP]:
        paddle2.move_ip(0, -paddle_speed)
    if keys[pygame.K_DOWN]:
        paddle2.move_ip(0, paddle_speed)

    # Boundary check for paddle1
    paddle1.y = max(20, min(height - 20 - paddle1.height, paddle1.y))

    # Boundary check for paddle2
    paddle2.y = max(20, min(height - 20 - paddle2.height, paddle2.y))

    # Move the ball
    ball_x += ball_speed_x
    ball_y += ball_speed_y

    # Boundary check for paddles
    paddle1.y = max(0, min(height - paddle1.height, paddle1.y))
    paddle2.y = max(0, min(height - paddle2.height, paddle2.y))

    # Boundary check for the ball
    if ball_y - ball_radius <= 20 or ball_y + ball_radius >= height - 20:
        ball_speed_y = -ball_speed_y

    # Collision check with paddles
    if paddle1.colliderect(pygame.Rect(ball_x - ball_radius, ball_y - ball_radius, ball_radius * 2, ball_radius * 2)):
        ball_speed_x = abs(ball_speed_x)  # Change direction
        current_ball_speed += speed_increment  # Increase the ball speed
        ball_speed_x *= current_ball_speed / ball_speed  # Update speed based on current_ball_speed

    if paddle2.colliderect(pygame.Rect(ball_x - ball_radius, ball_y - ball_radius, ball_radius * 2, ball_radius * 2)):
        ball_speed_x = -abs(ball_speed_x)  # Change direction
        current_ball_speed += speed_increment  # Increase the ball speed
        ball_speed_x *= current_ball_speed / ball_speed  # Update speed based on current_ball_speed

    # Check if ball hit the left or right wall
    if ball_x - ball_radius <= 0 or ball_x + ball_radius >= width:
        # Reset ball position to the center
        ball_x, ball_y = width // 2, height // 2

        if ball_x - ball_radius <= 0:
            paddle1_score += 1
        else:
            paddle2_score += 1

        # Randomize the angle again for the next movement
        if ball_direction == 0:
            ball_direction = 1
            ball_angle = math.radians(random.uniform(180-45, 180+45))  # Random angle between -45 and 45 degrees
        elif ball_direction == 1:
            ball_direction = 0
            ball_angle = math.radians(random.uniform(-45, 45))  # Random angle between -45 and 45 degrees
        current_ball_speed = 5
        ball_speed_x = current_ball_speed * math.cos(ball_angle)
        ball_speed_y = current_ball_speed * math.sin(ball_angle)
            
    # Clear the screen
    screen.fill(black)

    rect1 = pygame.draw.rect(screen, white, paddle1)
    rect2 = pygame.draw.rect(screen, white, paddle2)

    pygame.draw.line(screen, white, (0, 20), (width, 20), width= 10)
    pygame.draw.line(screen, white, (0, height-20), (width, height-20), width= 10)

    draw_line_dashed(screen, white, (width//2, 20), (width//2, height-20), width=10)

    font = pygame.font.SysFont("Arial", 36)
    txtsurf = font.render(str(paddle1_score), True, white)
    screen.blit(txtsurf,((width - txtsurf.get_width()) // 2 - 100, (height - txtsurf.get_height()) // 2))

    txtsurf = font.render(str(paddle2_score), True, white)
    screen.blit(txtsurf,((width - txtsurf.get_width()) // 2 + 100, (height - txtsurf.get_height()) // 2))

    pygame.draw.circle(screen, white, (ball_x, ball_y), ball_radius)

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    clock.tick(fps)