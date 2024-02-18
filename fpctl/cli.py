import typer

from . import dataset, feature, fingertip

app = typer.Typer()
app.add_typer(fingertip.app, name="fingertip")
app.add_typer(feature.app, name="feature")
app.add_typer(dataset.app, name="dataset")

if __name__ == "__main__":
    app()
