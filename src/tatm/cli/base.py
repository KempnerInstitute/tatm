import click
from tatm.cli.data import data


@click.group()
def cli():
    pass


cli.add_command(data)


if __name__ == "__main__":
    cli()
