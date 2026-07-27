// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "RinGo",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .library(name: "RinGoCore", targets: ["RinGoCore"]),
        .library(name: "RinGoModel", targets: ["RinGoModel"]),
        .library(name: "RinGoTrain", targets: ["RinGoTrain"]),
        .library(name: "RinGoEngine", targets: ["RinGoEngine"]),
        .executable(name: "ringo", targets: ["ringo"]),
    ],
    dependencies: [
        // Pinned exactly: mlx-swift >= 0.31.5 requires swift-tools-version 6.3, which exceeds the
        // Swift 6.2.4 Command Line Tools installed on this machine. Bump this once Xcode/toolchain
        // supports 6.3, then re-run `make lock`.
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.4"),
    ],
    targets: [
        .systemLibrary(name: "CZlib"),
        .target(name: "RinGoCore"),
        .target(
            name: "RinGoModel",
            dependencies: [
                "CZlib",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "RinGoEngine",
            dependencies: [
                "RinGoCore",
                "RinGoModel",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "RinGoTrain",
            dependencies: [
                "RinGoModel",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "ringo",
            dependencies: [
                "RinGoCore",
                "RinGoModel",
                "RinGoEngine",
                "RinGoTrain",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "RinGoCoreTests",
            dependencies: ["RinGoCore"],
            resources: [.copy("Fixtures"), .copy("Goldens")]
        ),
        .testTarget(
            name: "RinGoModelTests",
            dependencies: [
                "RinGoModel",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            resources: [.process("Fixtures")]
        ),
        .testTarget(
            name: "RinGoEngineTests",
            dependencies: ["RinGoEngine", "ringo"],
            resources: [.copy("GoldenTraces")]
        ),
        .testTarget(
            name: "RinGoTrainTests",
            dependencies: [
                "RinGoTrain",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
    ]
)
