import click

from tatm.cli.data import data
from tatm.cli.run import run
from tatm.cli.tokenizer import tokenize


@click.group()
def cli():
    pass


cli.add_command(data)
cli.add_command(tokenize)
cli.add_command(run)


if __name__ == "__main__":
    cli()
