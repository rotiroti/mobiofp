import typer

from . import dataset, feature, fingerprint, fingertip, quality

app = typer.Typer(help="Fingerphoto Recognition Command Line Interface.")
app.add_typer(fingertip.app, name="fingertip", help="Fingertip Commands")
app.add_typer(feature.app, name="feature", help="Feature Commands")
app.add_typer(dataset.app, name="dataset", help="Dataset Commands")
app.add_typer(quality.app, name="quality", help="Image Quality Commands")
app.add_typer(fingerprint.app, name="fingerprint", help="Fingerprint Commands")

if __name__ == "__main__":
    app()
