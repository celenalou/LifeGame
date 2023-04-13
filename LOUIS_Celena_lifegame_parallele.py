"""
Le jeu de la vie
################
Le jeu de la vie est un automate cellulaire inventé par Conway se basant normalement sur une grille infinie
de cellules en deux dimensions. Ces cellules peuvent prendre deux états :
    - un état vivant
    - un état mort
A l'initialisation, certaines cellules sont vivantes, d'autres mortes.
Le principe du jeu est alors d'itérer de telle sorte qu'à chaque itération, une cellule va devoir interagir avec
les huit cellules voisines (gauche, droite, bas, haut et les quatre en diagonales.) L'interaction se fait selon les
règles suivantes pour calculer l'irération suivante :
    - Une cellule vivante avec moins de deux cellules voisines vivantes meurt ( sous-population )
    - Une cellule vivante avec deux ou trois cellules voisines vivantes reste vivante
    - Une cellule vivante avec plus de trois cellules voisines vivantes meurt ( sur-population )
    - Une cellule morte avec exactement trois cellules voisines vivantes devient vivante ( reproduction )

Pour ce projet, on change légèrement les règles en transformant la grille infinie en un tore contenant un
nombre fini de cellules. Les cellules les plus à gauche ont pour voisines les cellules les plus à droite
et inversement, et de même les cellules les plus en haut ont pour voisines les cellules les plus en bas
et inversement.

On itère ensuite pour étudier la façon dont évolue la population des cellules sur la grille.
"""
import tkinter as tk
import numpy   as np
from mpi4py import MPI
from math import *

class grille:
    """
    Grille torique décrivant l'automate cellulaire.
    En entrée lors de la création de la grille :
        - dimensions est un tuple contenant le nombre de cellules dans les deux directions (nombre lignes, nombre colonnes)
        - init_pattern est une liste de cellules initialement vivantes sur cette grille (les autres sont considérées comme mortes)
        - color_life est la couleur dans laquelle on affiche une cellule vivante
        - color_dead est la couleur dans laquelle on affiche une cellule morte
    Si aucun pattern n'est donné, on tire au hasard quels sont les cellules vivantes et les cellules mortes
    Exemple :
       grid = grille( (10,10), init_pattern=[(2,2),(0,2),(4,2),(2,0),(2,4)], color_life="red", color_dead="black")
    """
    def __init__( self, dim, init_pattern = None, rank=0, p=1, q=1, Nx=0, Ny=0, mod_x=0, mod_y=0, color_life = "black", color_dead = "white" ):
        import random
        # Création d'indice (I, J), (p, q), (Nx, Ny), (modx, mody) pour chaque sous grille
        self.p = p
        self.q = q
        self.I = rank//q
        self.J = rank%q
        self.Nx = Nx
        self.Ny = Ny
        self.modx = mod_x
        self.mody = mod_y
        self.dimensions = (int(dim[0]), int(dim[1]))
        # Création de cells
        if init_pattern is not None:
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)
            if len(init_pattern) != 0:
                indices_i = np.array([], dtype=np.uint8)
                indices_j = np.array([], dtype=np.uint8)
                for v in init_pattern:
                    if self.I < mod_y:
                        indices_i = np.append(indices_i, v[0]-self.I*dim[0])
                    if self.I >= mod_y:
                        indices_i = np.append(indices_i, v[0]-self.I*dim[0]-mod_y)
                    if self.J < mod_x:
                        indices_j = np.append(indices_j, v[1]-self.J*dim[1])
                    if self.J >= mod_x:
                        indices_j = np.append(indices_j, v[1]-self.J*dim[1]-mod_x)
                self.cells[indices_i, indices_j] = 1
        else:
            self.cells = np.random.randint(2, size=dim, dtype=uint8)
        self.col_life = color_life
        self.col_dead = color_dead

    def compute_next_iteration(self):
        """
        Calcule la prochaine génération de cellules en suivant les règles du jeu de la vie
        """
        # Remarque 1: on pourrait optimiser en faisant du vectoriel, mais pour plus de clarté, on utilise les boucles
        # Remarque 2: on voit la grille plus comme une matrice qu'une grille géométrique. L'indice (0,0) est donc en haut
        #             à gauche de la grille !
        ny = self.dimensions[0]
        nx = self.dimensions[1]
        I = self.I
        J = self.J
        p = self.p
        q = self.q
        next_cells = np.empty(self.dimensions, dtype=np.uint8)
        diff_cells = []

        # Envoi
        comm.send(self.cells[0, :], dest=q*((I-1+p)%p)+J) # ligne haut
        comm.send(self.cells[ny-1, :], dest=q*((I+1)%p)+J) # ligne bas
        comm.send(self.cells[:, 0], dest=q*I+(J-1+q)%q) # ligne gauche
        comm.send(self.cells[:, nx-1], dest=q*I+(J+1)%q) # ligne droite
        comm.send(self.cells[0, 0], dest=q*((I-1+p)%p)+((J-1+q)%q)) # coin haut gauche
        comm.send(self.cells[0, nx-1], dest=q*((I-1+p)%p)+((J+1)%q)) # coin haut droit
        comm.send(self.cells[ny-1, 0], dest=q*((I+1)%p)+((J-1+q)%q)) # coin bas gauche
        comm.send(self.cells[ny-1, nx-1], dest=q*((I+1)%p)+((J+1)%q)) # coin bas droite

        # Reception
        ligne_bas = np.empty(nx, dtype=int)
        ligne_bas = comm.recv(source=q*((I+1)%p)+J) # ligne du dessous
        ligne_haut = np.empty(nx, dtype=int)
        ligne_haut = comm.recv(source=q*((I-1+p)%p)+J) # ligne du dessus
        ligne_droite = np.empty(ny, dtype=int)
        ligne_droite = comm.recv(source=q*I+((J+1)%q)) # ligne de droite
        ligne_gauche = np.empty(ny, dtype=int)
        ligne_gauche = comm.recv(source=q*I+((J-1+q)%q)) # ligne de gauche
        coin_bd = comm.recv(source=q*((I+1)%p)+((J+1)%q)) # coin bas droite
        coin_bg = comm.recv(source=q*((I+1)%p)+((J-1+q)%q)) # coin bas gauche
        coin_hd = comm.recv(source=q*((I-1+p)%p)+((J+1)%q)) # coin haut droite
        coin_hg = comm.recv(source=q*((I-1+p)%p)+((J-1+q)%q)) # coin haut gauche

        # Création sous grille avec les bords
        ug_vois = np.zeros((ny+2, nx+2), dtype=np.uint8)
        ug_vois[0, 1:nx+1] = ligne_haut
        ug_vois[ny+1, 1:nx+1] = ligne_bas
        ug_vois[1:ny+1, 0] = ligne_gauche
        ug_vois[1:ny+1, nx+1] = ligne_droite
        ug_vois[0, 0] = coin_hg
        ug_vois[0, nx+1] = coin_hd
        ug_vois[ny+1, 0] = coin_bg
        ug_vois[ny+1, nx+1] = coin_bd
        ug_vois[1:ny+1, 1:nx+1] = self.cells

        # Creation voisin
        voisins = [-1 for i in range(9)]
        for i in range(0, ny):
            for j in range(0, nx):
                voisins = np.concatenate(ug_vois[i:i+3, j:j+3])
                voisins = np.delete(voisins, 4)

                # Calcul de l'itération suivante
                nb_voisines_vivantes = np.sum(voisins)
                i_glob = I*self.Ny + min(I, self.mody) + i
                j_glob = J*self.Nx + min(J, self.modx) + j
                num_grille = i_glob*(q*Nx + mod_x) + j_glob
                if self.cells[i,j] == 1: # Si la cellule est vivante
                    if (nb_voisines_vivantes < 2) or (nb_voisines_vivantes > 3):
                        next_cells[i,j] = 0 # Cas de sous ou sur population, la cellule meurt
                        diff_cells.append(num_grille)
                    else:
                        next_cells[i,j] = 1 # Sinon elle reste vivante
                elif nb_voisines_vivantes == 3: # Cas où cellule morte mais entourée exactement de trois vivantes
                    next_cells[i,j] = 1         # Naissance de la cellule
                    diff_cells.append(num_grille)
                else:
                    next_cells[i,j] = 0         # Morte, elle reste morte.
        self.cells = next_cells
        return [diff_cells, next_cells]

class App:
    """
    Cette classe décrit la fenêtre affichant la grille à l'écran
        - geometry est un tuple de deux entiers donnant le nombre de pixels verticaux et horizontaux (dans cet ordre)
        - grid est la grille décrivant l'automate cellulaire (voir plus haut)
    """
    def __init__(self, geometry, grid):
        self.grid = grid
        # Calcul de la taille d'une cellule par rapport à la taille de la fenêtre et de la grille à afficher :
        self.size_x = geometry[1]//grid.dimensions[1]
        self.size_y = geometry[0]//grid.dimensions[0]
        if self.size_x > 4 and self.size_y > 4 :
            self.draw_color='black'
        else:
            self.draw_color=""
        # Ajustement de la taille de la fenêtre pour bien fitter la dimension de la grille
        self.width = grid.dimensions[1] * self.size_x
        self.height= grid.dimensions[0] * self.size_y
        # Création de la fenêtre à l'aide de tkinter
        self.root = tk.Tk()
        # Création de l'objet d'affichage
        self.canvas = tk.Canvas(self.root, height = self.height, width = self.width)
        self.canvas.pack()
        #
        self.canvas_cells = []

    def compute_rectange(self, i : int, j : int):
        """
        Calcul la géométrie du rectangle correspondant à la cellule (i,j)
        """
        return (self.size_x*j,self.height - self.size_y*i - 1, self.size_x*j+self.size_x-1, self.height - self.size_y*(i+1) )

    def compute_color(self,i : int,j : int):
        if self.grid.cells[i,j] == 0:
            return self.grid.col_dead
        else:
            return self.grid.col_life

    def draw(self, diff):
        if len(self.canvas_cells) == 0:
            # Création la première fois des cellules en tant qu'entité graphique :
            self.canvas_cells = [self.canvas.create_rectangle(*self.compute_rectange(i,j), fill=self.compute_color(i,j),outline=self.draw_color) for i in range(self.grid.dimensions[0]) for j in range(self.grid.dimensions[1])]
        else:
            nx = self.grid.dimensions[1]
            ny = self.grid.dimensions[0]
            [self.canvas.itemconfig(self.canvas_cells[ind], fill=self.compute_color(ind//nx,ind%nx),outline=self.draw_color) for ind in diff]
        self.root.update_idletasks()
        self.root.update()

if __name__ == '__main__':
    import time
    import sys
    dico_patterns = { # Dimension et pattern dans un tuple
        'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
        'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat" : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider": ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
        "glider_gun": ((200,100),[(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),(54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),(55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),(56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
        "space_ship": ((25,25),[(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
        "die_hard" : ((100,100), [(51,57),(52,51),(52,52),(53,52),(53,56),(53,57),(53,58)]),
        "pulsar": ((17,17),[(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),(2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),(4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),(10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
        "floraison" : ((40,40), [(19,18),(19,19),(19,20),(20,17),(20,19),(20,21),(21,18),(21,19),(21,20)]),
        "block_switch_engine" : ((400,400), [(201,202),(201,203),(202,202),(202,203),(211,203),(212,204),(212,202),(214,204),(214,201),(215,201),(215,202),(216,201)]),
        "u" : ((200,200), [(101,101),(102,102),(103,102),(103,101),(104,103),(105,103),(105,102),(105,101),(105,105),(103,105),(102,105),(101,105),(101,104)]),
        "flat" : ((200,400), [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200), (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),(106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),(114,200),(115,200),(116,200),(117,200),(118,200)])
    }
    choice = 'floraison'
    if len(sys.argv) > 1 :
        choice = sys.argv[1]
    resx = 800
    resy = 800
    if len(sys.argv) > 3 :
        resx = int(sys.argv[2])
        resy = int(sys.argv[3])

    init_pattern = dico_patterns[choice]

    # mpi4py
    comm_couple = MPI.COMM_WORLD.Dup()
    rank_couple = comm_couple.rank
    size_couple = comm_couple.size

    # Séparation entre le processus qui affiche et les processus qui calculent
    if rank_couple == 0 :
    	status_couple = 0
    else :
    	status_couple = 1
    comm = comm_couple.Split(status_couple, rank_couple)
    rank = comm.rank
    size = comm.size

    # Définition grille pour affichage
    if rank_couple == 0:
        grid = grille(*init_pattern, Nx=init_pattern[0][1], Ny=init_pattern[0][0], color_life='red')
        appli = App((resx,resy),grid)
        print(f"Pattern initial choisi : {choice}")
        print(f"resolution ecran : {resx,resy}")

    # Définition grilles de calcul
    if rank_couple != 0:
        # Définition p et q
        grand = max(init_pattern[0][:])
        q = max(gcd(i, size) for i in range(2, size))
        if q == 1:
            print("Le nombre de processus pour le calcul est premier")
            raise ValueError("Impossible de diviser la grille en sous grille")
        p = (size) // q
        if grand != init_pattern[0][0]:
            # Echange de p et q
            (p, q) = (q, p)

        # Définition taille sous grille
        Nx = init_pattern[0][0] // q
        mod_x = init_pattern[0][0] % q
        size_rect_x = Nx * np.ones(q)
        if mod_x != 0:
            for i in range(mod_x):
                size_rect_x[i] += 1
        Ny = init_pattern[0][1] // p
        mod_y = init_pattern[0][1] % p
        size_rect_y = Ny * np.ones(p)
        if mod_y != 0:
            for i in range(mod_y):
                size_rect_y[i] += 1
        I = rank//q
        J = rank%q
        x = int(size_rect_y[I])
        y = int(size_rect_x[J])
        taille_ug = (x, y)

        # Pour donner init_pattern dans le self.cells (oui c'est pas du tout opti oops)
        ligne = [0 for i in range(len(init_pattern[1]))]
        colonne = [0 for i in range(len(init_pattern[1]))]
        i = 0
        for elt in init_pattern[1]:
            if mod_y == 0:
                ligne[i] = elt[0]//x
            elif I < mod_y:
                ligne[i] = elt[0]//x
            elif I >= mod_y:
                ligne[i] = elt[0]//(x+1)
            if mod_x == 0:
                colonne[i] = elt[1]//y
            elif J < mod_x:
                colonne[i] = elt[1]//y
            elif J >= mod_x:
                colonne[i] = elt[1]//(y+1)
            i += 1
        sous_elt = []
        for i in range(len(ligne)):
            if I == ligne[i] and J == colonne[i]:
                sous_elt.append(init_pattern[1][i])
        under_grid = grille(taille_ug, sous_elt, rank, p, q, Nx, Ny, mod_x, mod_y)

    # Boucle pour l'affichage
    #while(True):
    for i in range(100):
        time.sleep(1) # A régler ou commenter pour vitesse maxi
        t1 = time.time()
        # Calcul de la prochaine iteration et envoi au rang 0 de comm_couple
        if rank_couple != 0:
            [diff, next_cells] = under_grid.compute_next_iteration()
            next_cells_glob = comm.gather(next_cells, root=0)
            diff_glob = comm.reduce(diff, root=0, op=MPI.SUM)
            if rank == 0:
                test_next = [0 for i in range(0, p)]
                for j in range(0, p):
                    test_next[j] = np.concatenate([next_cells_glob[i+j*q] for i in range(0, q)], axis=1)
                next_cells_glob = np.concatenate([test_next[j] for j in range(0, p)], axis=0)
                comm_couple.send([diff_glob, next_cells_glob], dest=0)
        # Réception de la prochaine itération et affichage
        else:
            diff_glob = []
            next_cells_glob = []
            [diff_glob, next_cells_glob] = comm_couple.recv(source=1)
            grid.cells = next_cells_glob
            t2 = time.time()
            appli.draw(diff_glob)
            t3 = time.time()
            print(f"Temps calcul prochaine generation : {t2-t1:2.2e} secondes, temps affichage : {t3-t2:2.2e} secondes\r", end='', flush=1);
