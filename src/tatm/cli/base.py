import click

from tatm.cli.data import data
from tatm.cli.tokenizer import tokenize


@click.group()
def cli():
    pass


cli.add_command(data)
cli.add_command(tokenize)


if __name__ == "__main__":
    cli()
