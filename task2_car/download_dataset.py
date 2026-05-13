import kagglehub


def main():
	path = kagglehub.dataset_download("ashfakyeafi/road-vehicle-images-dataset")
	print(f"Dataset downloaded to: {path}")


if __name__ == "__main__":
	main()