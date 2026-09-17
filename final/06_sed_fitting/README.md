# prospector

Before running this stage, generate the flux and error files using the scripts in
`final/05_photometry/`.

## Command-Line Options

Run `prospector_code.py` from the repository root. Options include:

| Option | Description |
| --- | --- |
| `--showplots` | Show plots from an existing or newly fitted model. |
| `--readfile FILE_PATH` | Specify the `.h5` file to inspect when plotting. |
| `--fitnewmodel` | Fit a new model. This can take a long time. |
| `--outfile FILE_PATH` | Specify the `.h5` output file for a new fit. |
| `--minimize` | Perform the optional optimization step before fitting. |

## Optional Email Notification

To enable email notifications, create a `.env` file in this folder containing:

```text
EMAIL=your-email-address
PASSWD=your-email-password
SERVER=your-smtp-server
```