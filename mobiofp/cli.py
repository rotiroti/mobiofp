import typer

from . import dataset, feature, fingertip, minutiae

app = typer.Typer()
app.add_typer(fingertip.app, name="fingertip")
app.add_typer(feature.app, name="feature")
app.add_typer(dataset.app, name="dataset")
app.add_typer(minutiae.app, name="minutiae")

if __name__ == "__main__":
    app()
