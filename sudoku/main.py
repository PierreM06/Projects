from collections import Counter

class cell():
    def __init__(self, number) -> None:
        self.number = number
        self.value = None
        self.possible = [1,2,3,4,5,6,7,8,9]
        pass

    def remove(self, waarde) -> None:
        if waarde in self.possible and len(self.possible) > 1:
            self.possible.remove(waarde)
            if len(self.possible) == 1:
                self.sett(self.possible[0])

    def sett(self, waarde) -> None:
        self.value = waarde
        if waarde != 0:
            self.possible = []
            self.update_group()

    def update_group(self) -> None:
        rows[int(self.number//9)].update_cells(self.value)
        columns[int(self.number%9)].update_cells(self.value)
        squares[int((self.number//27*3))+int(self.number%9/3)].update_cells(self.value)


class row():
    def __init__(self, cells: list[cell]) -> None:
        self.possible = [1,2,3,4,5,6,7,8,9]
        self.containing = cells
        pass

    def update_cells(self, waarde) -> None:
        if waarde in self.possible:
            self.possible.remove(waarde)
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
                        i.sett(key)


class column():
    def __init__(self, cells: list[cell]) -> None:
        self.possible = [1,2,3,4,5,6,7,8,9]
        self.containing = cells
        pass

    def update_cells(self, waarde) -> None:
        if waarde in self.possible:
            self.possible.remove(waarde)
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
                        i.sett(key)


class square():
    def __init__(self, cells: list[cell]) -> None:
        self.possible = [1,2,3,4,5,6,7,8,9]
        self.containing = cells
        pass

    def update_cells(self, waarde) -> None:
        if waarde in self.possible:
            self.possible.remove(waarde)
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
                        i.sett(key)


def grid():
    getallen = []
    for i in cells:
        if i.value is None:
            getallen.append(0)
        getallen.append(i.value)
    x=1
    for i in range(9):
        print(f"{getallen[i*9:i*9+3]}|{getallen[i*9+3:i*9+6]}|{getallen[i*9+6:i*9+9]}")
        if x%3 == 0 and x != 9:
            print("---------+---------+---------")
        x += 1
    
cells = [cell(i) for i in range(9*9)]

rows = [row([cells[i+j*9] for i in range(9)]) for j in range(9)]

columns = [column([cells[i*9+j] for i in range(9)]) for j in range(9)]

squares = [square([cells[i%3+(i//3*9)+(j%3)*3+(j//3)*27] for i in range(9)]) for j in range(9)]

eigen_regel = "9000500003000000790250900300602090000003085146040630290100964080870003924450872610"
for i in range(81):
    cells[i].sett(int(eigen_regel[i]))

for i in range(100):
    done = True

    for j in cells:
        if j.value != 0:
            j.update_group()
        else:
            done = False

    for r in rows:
        r.group()
    for c in columns:
        c.group()
    for s in squares:
        s.group()
    
    if done:
        print("klaar")
        break

grid()

leeg = [i for i in cells if len(i.possible) == 0 and i.value == 0]
print(len(leeg))
