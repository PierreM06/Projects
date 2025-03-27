import pyautogui
import time
from PIL import Image
import random

# Define colors for the squares and numbers
square_colors = {
    ((254, 254, 254), (255, 255, 255)): '🟩',
    ((189, 189, 189), (190, 190, 190)): '⬜️',
    ((189, 68, 64), (189, 69, 64)): '❌',
    ((0, 0, 254), (1, 1, 255)): ' 1',
    ((199, 192, 200), (199, 192, 200)): ' 2',
    ((185, 201, 201), (185, 201, 201)): ' 3',
    ((128, 0, 128), (128, 0, 128)): ' 4'

}

# Function to capture the Minesweeper board from a screenshot with a delay
def capture_board_with_delay(start_x, start_y, width, height, num_cells_x, num_cells_y):
    print("Move your mouse to the Minesweeper board...")
    screenshot = pyautogui.screenshot(region=(start_x, start_y, width, height))
    screenshot.save("board.png")
    cell_width = width // num_cells_x
    cell_height = height // num_cells_y
    return screenshot, cell_width, cell_height

# Function to identify cells and their status
def identify_cells(board, cell_width, cell_height, num_cells_x, num_cells_y):
    # Convert the PIL Image to RGB mode
    board = board.convert('RGB')
    pixels = board.load()
    cell_status = []

    # Iterate over each cell
    for y in range(num_cells_y):
        row = []
        for x in range(num_cells_x):
            # Get the RGB values of the center of the cell
            pixel_color = pixels[x * cell_width + cell_width / 2, 
                                 y * cell_height + cell_height / 2-3]
            # Check if it matches any predefined colors
            status = None
            for color_range, symbol in square_colors.items():
                r_in_range = color_range[0][0] <= pixel_color[0] <= color_range[1][0]
                g_in_range = color_range[0][1] <= pixel_color[1] <= color_range[1][1]
                b_in_range = color_range[0][2] <= pixel_color[2] <= color_range[1][2]
                if r_in_range and g_in_range and b_in_range:
                    status = symbol
                    break
            if status is None:
                status = pixel_color
                # print(f"{status}-({x},{y}).png")
                # board.save(f"wrong/{status}-({x},{y}).png")
            row.append(status)
        cell_status.append(row)

    return cell_status

# Define a function to check if the game is won
def victory(board):
    for row in board:
        for cell in row:
            # If there are any unrevealed cells that are not mines, return False
            if cell == '🟩':
                return False
    return True

def check_neighbors(board, x, y, number):
    greens, flags = [], []
    num_rows = len(board)
    num_cols = len(board[0])

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if 0 <= y + dy < num_rows and 0 <= x + dx < num_cols:
                neighbor = board[y + dy][x + dx]
                if neighbor == "🟩":
                    greens.append((y + dy, x + dx))
                elif neighbor == "❌":
                    flags.append((y + dy, x + dx))

    if number == len(flags):
        return [], greens
    elif number == len(greens) + len(flags):
        return greens, []
    return [], []

def green_magic(board):
    return

# Define a function to make a move
def make_move(board):
    right_clicks, left_clicks = [], []

    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            if cell in [" 1", " 2", " 3", " 4", " 5", " 6"]:
                number = int(cell)
                right, left = check_neighbors(board, x, y, number)
                for click in right:
                    right_clicks.append(click)
                for click in left:
                    left_clicks.append(click)

    r, l = [], []
    for i in right_clicks:
        if i not in r and i != []:
            r.append(i)
    for i in left_clicks:
        if i not in l and i != []:
            l.append(i)

    if r == [] and l == []:
        green_magic(board)
    return r, l

# Main function
def main():
    start_x = 15
    start_y = 181

    # easy minesweeper spel
    width = 270
    height = 270
    num_cells_x = 9
    num_cells_y = 9

    # medium minesweeper spel
    # width = 480
    # height = 480
    # num_cells_x = 16
    # num_cells_y = 16

    # # groot minesweeper spel
    # width = 600
    # height = 500
    # num_cells_x = 24
    # num_cells_y = 20

    time.sleep(1)
    pyautogui.click(random.randint(start_x, start_x+width), random.randint(start_y, start_y + height))
    time.sleep(1)

    # Capture the Minesweeper board from the screenshot with a delay
    captured_board, cell_width, cell_height = capture_board_with_delay(start_x, start_y, width, height, num_cells_x, num_cells_y)

    while True:
        # Identify cells and their status
        board_with_cells = identify_cells(captured_board, cell_width, cell_height, num_cells_x, num_cells_y)
        for row in board_with_cells:
            print(*row, sep="")
        print()

        break

        # Make a move
        right, left = make_move(board_with_cells)

        # Click the cell on the screen
        for y,x in right:
            pyautogui.rightClick(start_x + x * cell_width + cell_width // 2, start_y + y * cell_height + cell_height // 2)
        for y,x in left:
            pyautogui.click(start_x + x * cell_width + cell_width // 2, start_y + y * cell_height + cell_height // 2)

        pyautogui.moveTo(200, 200)
        # time.sleep(.2)
        # Take another screenshot
        captured_board = pyautogui.screenshot(region=(start_x, start_y, width, height))

        if victory(board_with_cells):
            break

    print("Game won!")

if __name__ == "__main__":
    main()

