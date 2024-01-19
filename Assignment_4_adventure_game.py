
# IMPORTANT NOTES   : do not forget to install these modules (notably "Pillow") if not if your environment !
#                   : place the image "map.png" in the same folder as this script
#                   : the function "match()" will only work well for Python users above Python >= 3.10 !


import random
import os
# from PIL import Image
import re as re

# STEP 1 : Initialising and defining components of the game, aka :
#       "items" (to be placed in the different rooms), 
#       "inventory" (with the items picked by the user), 
#       "description" (of the game rooms), 
#       "rooms" with different exits, 
#       "options" for the user to select after each move
#       "Player", a class allowing to store/change location of the user


items = ["diamond ring", "key", "knife", "trap", "russian tank", "rose"]

inventory = []

description = {
    "Entrance": "You are in the house entrance. There is a King Charles III painting, a (stolen) Banksy, a selfie stick, and doors to the north and east",
    "Garage": "You are in a garage. Theres a new Tesla, a Benz and a Russian Tank. Also a door to the west and north.",
    "Hallway": "You are in a hallway. There are doors to the north, east, west and south.",
    "Garden": "You are in a Garden. There are roses fertilizer and a corpse, and a door to the west and south,",
    "Master Bedroom": "You are in a bedroom. There is a bed, a dresser and a key. There's a door to the south and a door to the west.",
    "Bathroom": "You are in a bathroom. There is a sink, a toilet and a diamond ring, and a door to the south and east.",
    "Kitchen":"You are in the kitchen there is a knife, a ham leg and fish and chips. There is a door to the east and to the north."
}

rooms = {
    "Entrance": {"east": "Garage", "north": "Hallway"},
    "Garage": {"west": "Entrance", "north": "Garden"},
    "Hallway": {"north": "Master Bedroom", "south": "Entrance", "east": "Garden", "west": "Kitchen"},
    "Garden": {"west": "Hallway", "south": "Garage"},
    "Master Bedroom": {"south": "Hallway", "west": "Bathroom"},
    "Bathroom": {"south": "Kitchen", "east": "Master Bedroom"},
    "Kitchen": {"east": "Hallway", "north": "Bathroom"}
}

# Allow options to user, notably "quit" and secret option "open door"
options = ["Move to a different room", "Find item", "quit", "Open door"]



class Player:

    def __init__(self, position: str):
        self.position = position

    def get_position(self):
        return self.position

    def set_position(self, position):
        self.position = position


# Instantiate player object
player = Player("Entrance")

# STEP 2 CREATING FUNCTIONS NECESSARY TO THE GAME FUNCTIONING

# clears the terminal after each move to make the game look "cleaner" (readable) after each move 
def clear():
    
    os.system("cls" if os.name == "nt" else "clear")


# Set the game "entrance menu", which the player sees when starting the game. Option to show map or not (NB : image should be in the same)
def map():
    
    print("Welcome to the Mansion!\n")
    print("Rules - To win the game you need to pick up 4 or more items and reach the 'Master Bedroom' to win the game.\n      - Be careful to not step on any traps or its game over!")
    main_menu = input("\nMap or no map? (y/n): ")

    while True:

        # User wants the map
        if main_menu == "y":
            img = Image.open('map.PNG')
            img.show()
            break
        
        # User wants no map
        elif main_menu == "n":
            break
        
        # Force user to make his mind
        else:
            print("Invalid input.")
            continue
    print("")


# Displays the options available to the player after every move
def prompt():
    
    print(f"Location: {player.get_position()}\n{description[player.get_position()]}\n")
    print(f"Inventory: {inventory}")
    print("-------------------------")
    
    # If user collects more than 4 items, give user an extra option !
    if len(inventory) >= 4 and player.get_position() == "Master Bedroom":
        
        for i in range(len(options)):
            print(f"{i + 1}. {options[i]}")
        print("-------------------------")
        print("")

    else:
        
        for i in range(len(options) - 1):
            print(f"{i + 1}. {options[i]}")
        print("-------------------------")
        print("")


# Main game loop :
def game_loop():

    # Randomizes item locations except for the item in the first room so the player fall into a trap instantly
    random.shuffle(items)
    item_location = {
    "Entrance": "",
    "Garage": items[0], 
    "Hallway": items[1],
    "Garden": items[2],
    "Master Bedroom": items[3],
    "Bathroom": items[4],
    "Kitchen": items[5]
    }


    while True:

        prompt()
        next_move = (input("Next move (Type 1, 2 or 3): "))

        re.match(next_move):

            # movement of player in specific direction
            case "1":

                while True:

                    direction = input("where to you want to move? (Type: north, west, east or south)\n")

                    # Move the user to one of these directions
                    if direction in ["north", "west", "east", "south"]:
                        clear()

                        try: 
                            player.set_position(rooms[player.get_position()][direction])
                            break

                        except KeyError:
                            print("Can't move in that direction.")
                    
                    
                    # Allows the user to go back to user menu if no move is desired
                    elif direction == "back":
                        break


                    # Forces the user to enter a correct input
                    else:
                        clear()
                        print("Invalid Input. Please enter 'north, west, east or south'.\n")
                        continue


            # Picking up item with "option 2"
            case "2":

                clear()

                # If the wrong item is selected :
                if item_location[player.get_position()] == "trap":
                    print("you have died.")
                    return False
                
                # Otherwise ...
                elif item_location[player.get_position()]:
                    print(f"you have found {item_location.get(player.get_position())}")
                    inventory.append(item_location.get(player.get_position()))
                    item_location[player.get_position()] = ""

                
                elif item_location[player.get_position()] == "":
                    print("You find nothing")
                    continue


            # Quitting the game if "option 3"
            case "3":
                break


            # Beating the game with secret "option 4"
            case "4":
                print("you have beaten the game!")
                break


            # Ensure the user enters correct input
            case _:
                clear()
                print("Please enter a valid input.\n")
                continue


# 
def main():
    clear()
    map()
    game_loop()

if __name__ == "__main__":
    main()