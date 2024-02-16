import typer

from . import dataset, feature, fingertip, template

app = typer.Typer()
app.add_typer(fingertip.app, name="fingertip")
app.add_typer(feature.app, name="feature")
app.add_typer(dataset.app, name="dataset")
app.add_typer(template.app, name="template")

if __name__ == "__main__":
    app()
