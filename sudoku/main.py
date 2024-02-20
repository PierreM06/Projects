from collections import Counter
import customtkinter as tk

class cell():
    def __init__(self, number) -> None:
        self.number = number
        self.value = None
        self.possible = [1,2,3,4,5,6,7,8,9]

        self.frame = tk.CTkFrame(root, width=65, height=65, border_color="#858585", border_width=1, corner_radius=0)
        for i in range(9):
            tk.CTkButton(self.frame, text=i+1, width=60/3, height=60/3, fg_color="transparent", 
                        command=lambda num=i+1: self.clicked(num), border_width=0, corner_radius=0,
                        text_color="#000000").grid(row=i//3, column=i%3)

        self.frame.grid(row=number//9, column=number%9, ipadx=1, ipady=1)
        pass

    def remove(self, waarde) -> None:
        if waarde in self.possible:
            self.possible.remove(waarde)
            self.update_label()
            if len(self.possible) == 1:
                self.clicked(self.possible[0])
        else:
            return

    def sett(self, waarde) -> None:
        self.value = waarde
        if waarde != 0:
            self.possible = []
            self.update_group()

    def update_group(self) -> None:
        rows[int(self.number//9)].update_cells(self.value)
        columns[int(self.number%9)].update_cells(self.value)
        squares[int((self.number//27*3))+int(self.number%9/3)].update_cells(self.value)

    def clicked(self, waarde):
        self.frame.destroy()

        self.label = tk.CTkLabel(root, text=waarde, width=55/9, height=610/9, text_color="#FFFFFF", font=tk.CTkFont(size=36))
        self.label.grid(row=self.number//9, column=self.number%9, padx=19)
        
        self.sett(waarde)

    def update_label(self):
        self.frame.destroy()

        self.frame = tk.CTkFrame(root, width=65, height=65, border_color="#858585", border_width=5, corner_radius=0)
        for i in range(9):
            if i+1 not in self.possible:
                tk.CTkButton(self.frame, text=" ", width=60/3, height=60/3, fg_color="transparent", border_width=0, corner_radius=0, hover=False).grid(row=i//3, column=i%3)
                continue
            tk.CTkButton(self.frame, text=i+1, width=60/3, height=60/3, fg_color="transparent", command=lambda num=i+1: self.clicked(num), border_width=0, corner_radius=0, text_color="#000000").grid(row=i//3, column=i%3)

        self.frame.grid(row=self.number//9, column=self.number%9, ipadx=1, ipady=1)
        pass


class group():
    def __init__(self, cells: list[cell]) -> None:
        self.containing = cells
        pass

    def update_cells(self, waarde) -> None:
        for i in self.containing:
            i.remove(waarde)
        
    def group(self) -> None:
        self.group_possible = []
        for i in self.containing:
            for j in i.possible:
                self.group_possible.append(j)
        self.voorkomen = Counter(self.group_possible)
        for key, value in self.voorkomen.items():
            if value == 1:
                for i in self.containing:
                    if key in i.possible:
                        i.clicked(key)


def grid():
    for i in range(9):
        print(f"{cells[i*9].value}{cells[i*9+1].value}{cells[i*9+2].value}{cells[i*9+3].value}{cells[i*9+4].value}{cells[i*9+5].value}{cells[i*9+6].value}{cells[i*9+7].value}{cells[i*9+8].value}")
    
root = tk.CTk()
root.geometry("555x610")
root.resizable(False, False)
    
cells = [cell(i) for i in range(9*9)]

v_line1 = tk.CTkLabel(root, text="", width=2, height=610, fg_color="#FFFFFF")
v_line2 = tk.CTkLabel(root, text="", width=2, height=610, fg_color="#FFFFFF")

h_line1 = tk.CTkLabel(root, font=tk.CTkFont(size=1), text="", width=560, height=2, fg_color="#FFFFFF")
h_line2 = tk.CTkLabel(root, font=tk.CTkFont(size=1), text="", width=560, height=2, fg_color="#FFFFFF")

v_line1.place(x=185, y=0)
v_line2.place(x=371, y=0)

h_line1.place(x=0, y=202)
h_line2.place(x=0, y=406)

rows = [group([cells[i+j*9] for i in range(9)]) for j in range(9)]

columns = [group([cells[i*9+j] for i in range(9)]) for j in range(9)]

squares = [group([cells[i%3+(i//3*9)+(j%3)*3+(j//3)*27] for i in range(9)]) for j in range(9)]

# regel = "200507406000031000000000230000020000860310000045000000009000700006950002001006008"
# print(len(regel))
# for i in range(81):
#     cells[i].sett(int(regel[i]))

# done = False
# while not done:
#     for j in cells:
#         if j.value != 0 or j.value is not None:
#             j.update_group()
#             done = True

#     for r in rows:
#         r.group()
#     for c in columns:
#         c.group()
#     for s in squares:
#         s.group()

#     time.sleep(1)

# grid()

# if len([i for i in cells if i.value == 0]) == 0: 
#     print("Done")
# else: 
#     print(len([i for i in cells if i.value == 0 and len(i.possible) == 0]))


root.mainloop()
