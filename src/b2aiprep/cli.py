import logging

import click

from b2aiprep.commands import (  # gensynthtabdata,
    batchconvert,
    convert,
    create_derived_dataset,
    createbatchcsv,
    dashboard,
    prepare_bids,
    publish_bids_dataset,
    redcap2bids,
    transcribe,
    validate,
    validate_derived_dataset,
    verify,
    reproschema_audio_to_folder,
    reproschema_to_redcap,
    generate_audio_features,
    bids2shadow,
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
cli.add_command(validate_derived_dataset)
cli.add_command(publish_bids_dataset)
cli.add_command(reproschema_audio_to_folder)
cli.add_command(reproschema_to_redcap)
cli.add_command(generate_audio_features)
cli.add_command(bids2shadow)

if __name__ == "__main__":
    # include main to enable python debugging
    cli()  # pylint: disable=no-value-for-parameter
