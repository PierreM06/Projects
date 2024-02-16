import pyautogui
import time
from icecream import ic


# Define colors for the squares and numbers
square_colors = {
    ((155, 195, 90), (200, 225, 140)): "🟩",
    ((190, 70, 35), (225, 145, 65)): "❌",
    ((210, 185, 157), (223, 195, 163)): "⬜️",
    ((50, 110, 195), (100, 150, 205)): " 1",
    ((100, 145, 85), (180, 180, 135)): " 2",
    ((195, 70, 60), (210, 160, 125)): " 3",
    ((120, 50, 155), (125, 60, 160)): " 4",
    ((210, 160, 85), (236, 190, 150)): " 5",
    ((90, 150, 160), (100, 165, 170)): " 6"
}

# Function to capture the Minesweeper board from a screenshot with a delay
def capture_board_with_delay(start_x, start_y, width, height, num_cells_x, num_cells_y, delay):
    print("Move your mouse to the Minesweeper board...")
    time.sleep(delay)
    screenshot = pyautogui.screenshot(region=(start_x, start_y, width, height))
    screenshot.save("board.png")
    cell_width = width // num_cells_x
    cell_height = height // num_cells_y
    return screenshot, cell_width, cell_height

# Function to identify cells and their status
def identify_cells(board, cell_width, cell_height, num_cells_x, num_cells_y, start_x, start_y):
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
                # board.save(f"{status}-({x},{y}).png")
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
    return r, l

# Main function
def main():
    # Define the coordinates and dimensions for capturing the Minesweeper board from the screenshot
    start_x = 615
    start_y = 415
    width = 450
    height = 360
    num_cells_x = 10  # Number of cells on the x-axis
    num_cells_y = 8   # Number of cells on the y-axis

    start_x = 570
    start_y = 385
    width = 540
    height = 420
    num_cells_x = 18
    num_cells_y = 14

    start_x = 540
    start_y = 345
    width = 600
    height = 500
    num_cells_x = 24
    num_cells_y = 20

    delay = 1  # 5 seconds delay (adjust as needed)

    # Capture the Minesweeper board from the screenshot with a delay
    captured_board, cell_width, cell_height = capture_board_with_delay(start_x, start_y, width, height, num_cells_x, num_cells_y, delay)

    while True:
        # Identify cells and their status
        board_with_cells = identify_cells(captured_board, cell_width, cell_height, num_cells_x, num_cells_y, start_x, start_y)
        for row in board_with_cells:
            print(*row, sep="")
        print()

        # Make a move
        right, left = make_move(board_with_cells)

        # Click the cell on the screen
        for y,x in right:
            pyautogui.rightClick(start_x + x * cell_width + cell_width // 2, start_y + y * cell_height + cell_height // 2)
        for y,x in left:
            pyautogui.click(start_x + x * cell_width + cell_width // 2, start_y + y * cell_height + cell_height // 2)

        time.sleep(.11)
        # Take another screenshot
        captured_board = pyautogui.screenshot(region=(start_x, start_y, width, height))

        if victory(board_with_cells):
            break

    print("Game won!")

if __name__ == "__main__":
    main()
