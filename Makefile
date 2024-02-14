CHART_PATH=infrastructure/helm/conformal-data-cleaning

check-docker-prerequisites:
ifndef DOCKER_IMAGE_NAME
	$(error DOCKER_IMAGE_NAME is not set)
endif

build-docker: check-docker-prerequisites
	poetry build
	rm -rf garf/dist
	mv dist garf/dist

	docker build -t ${DOCKER_IMAGE_NAME}:final -f infrastructure/docker/Dockerfile .
	docker build -t ${DOCKER_IMAGE_NAME}:garf -f infrastructure/docker/Dockerfile.garf garf

push-docker: check-docker-prerequisites
	docker push ${DOCKER_IMAGE_NAME}:final
	docker push ${DOCKER_IMAGE_NAME}:garf

docker: build-docker push-docker

helm-delete:
	# ignoring error as long as it does not exist
	-helm delete $(shell helm list --filter conformal-data-cleaning --short)
	-helm delete $(shell helm list --filter conformal-data-cleaning-garf --short)

helm-install:
	cd scripts && python deploy_experiments.py

deploy-all: docker helm-install