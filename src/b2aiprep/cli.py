import logging

import click

from b2aiprep.commands import (  # gensynthtabdata,
    batchconvert,
    bids2shadow,
    convert,
    create_bundled_dataset,
    createbatchcsv,
    dashboard,
    generate_audio_features,
    redcap2bids,
    reproschema_to_redcap,
    transcribe,
    validate_phenotype_command,
    validate_bundled_dataset,
    validate_feature_extraction,
    verify,
    deidentify_bids_dataset,
    run_quality_control_on_audios,
    update_bids_template_command,
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
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


cli.add_command(dashboard)
cli.add_command(redcap2bids)
# cli.add_command(gensynthtabdata)
cli.add_command(convert)
cli.add_command(batchconvert)
cli.add_command(verify)
cli.add_command(transcribe)
cli.add_command(createbatchcsv)
cli.add_command(create_bundled_dataset)
cli.add_command(validate_phenotype_command)
cli.add_command(validate_bundled_dataset)
cli.add_command(deidentify_bids_dataset)
cli.add_command(reproschema_to_redcap)
cli.add_command(generate_audio_features)
cli.add_command(bids2shadow)
cli.add_command(validate_feature_extraction)
cli.add_command(run_quality_control_on_audios)
cli.add_command(update_bids_template_command)

if __name__ == "__main__":
    # include main to enable python debugging
    cli()  # pylint: disable=no-value-for-parameter
