import typer

from . import dataset, feature, fingertip

app = typer.Typer(help="Fingerphoto Recognition Command Line Interface.")
app.add_typer(fingertip.app, name="fingertip", help="Fingertip Commands")
app.add_typer(feature.app, name="feature", help="Feature Commands")
app.add_typer(dataset.app, name="dataset", help="Dataset Commands")

if __name__ == "__main__":
    app()
