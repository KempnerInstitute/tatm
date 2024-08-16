import click

from tatm.data.metadata import create_metadata_interactive


@click.group()
def data():
    pass


@data.command()
def create_metadata():
    create_metadata_interactive()
