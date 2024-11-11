import logging

import click

from b2aiprep.commands import (  # gensynthtabdata,
    batchconvert,
    convert,
    create_derived_dataset,
    createbatchcsv,
    dashboard,
    prepare_bids,
    redcap2bids,
    transcribe,
    validate,
    verify,
)


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the log level for the CLI.",
)
@click.pass_context
def cli(ctx, log_level):
    ctx.ensure_object(dict)
    ctx.obj["LOG_LEVEL"] = log_level

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    _LOGGER = logging.getLogger(__name__)


cli.add_command(dashboard)
cli.add_command(redcap2bids)
cli.add_command(prepare_bids)
cli.add_command(validate)
# cli.add_command(gensynthtabdata)
cli.add_command(convert)
cli.add_command(batchconvert)
cli.add_command(verify)
cli.add_command(transcribe)
cli.add_command(createbatchcsv)
cli.add_command(create_derived_dataset)

if __name__ == "__main__":
    # include main to enable python debugging
    cli()  # pylint: disable=no-value-for-parameter
