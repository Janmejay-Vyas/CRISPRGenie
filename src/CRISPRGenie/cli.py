import typer

app = typer.Typer()


@app.command()
def greet(name: str = typer.Option("World", help="Name to greet")):
    """
    Greets a user with the provided name.
    :param name: Name to greet
    :type name: str
    :return: None
    """
    typer.echo(f"Hello {name}!")


@app.command()
def add(
    a: float = typer.Argument(..., help="First number"),
    b: float = typer.Argument(..., help="Second number"),
):
    """
    Adds two numbers and prints the result.
    :param a: First number
    :type a: float
    :param b: Second number
    :type b: float
    :return: None
    """
    result = a + b
    typer.echo(f"The sum of {a} and {b} is {result}")


if __name__ == "__main__":
    app()
