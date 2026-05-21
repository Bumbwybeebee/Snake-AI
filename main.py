import snake
import Apple
import ai

def main():
    res = 15
    print("start")
    player_snake = snake.Snake(res)
    apple = Apple.Apple(res)

    try:
        while True:
            # main loop
            #eating
            if(player_snake.snake_head == apple.applePos):
                player_snake.grow
                apple.generate(player_snake)
            pass
    except KeyboardInterrupt:
        print("Program stopped by user.")

if __name__ == "__main__":
    main()