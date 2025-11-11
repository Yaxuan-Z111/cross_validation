cross_validation
================
yz5248
2025-11-11

``` r
data("lidar")

lidar_df = 
  lidar |> 
  as_tibble() |>
  mutate(id = row_number())

lidar_df |> 
  ggplot(aes(x = range, y = logratio)) + 
  geom_point()
```

![](cross_validation_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

# create dataframe

``` r
train_df = 
  sample_frac(lidar_df, size = .8) |>
  arrange(id)

test_df = anti_join(lidar_df, train_df, by = "id")
```

``` r
linear_mod = lm(logratio ~ range, data = train_df)
```

``` r
train_df |>
  add_predictions(linear_mod) |>
  ggplot(aes(x = range, y = logratio)) +
  geom_point() +
  geom_line(aes(y = pred), color = "red")
```

![](cross_validation_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
smooth_mod = mgcv::gam(logratio ~ s(range), data = train_df)
```

``` r
train_df |>
  add_predictions(smooth_mod) |>
  ggplot(aes(x = range, y = logratio)) +
  geom_point() +
  geom_line(aes(y = pred), color = "red")
```

![](cross_validation_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
wiggly_mod = mgcv::gam(logratio ~ s(range, k = 30), sp = 10e-6, data = train_df)
```

``` r
train_df |>
  add_predictions(wiggly_mod) |>
  ggplot(aes(x = range, y = logratio)) +
  geom_point() +
  geom_line(aes(y = pred), color = "red")
```

![](cross_validation_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
wiggly_mod2 = mgcv::gam(logratio ~ s(range, k = 50), sp = 10e-8, data = train_df)
```

``` r
train_df |>
  add_predictions(wiggly_mod2) |>
  ggplot(aes(x = range, y = logratio)) +
  geom_point() +
  geom_line(aes(y = pred), color = "red")
```

![](cross_validation_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
train_df |> 
  gather_predictions(linear_mod, smooth_mod, wiggly_mod) |> 
  mutate(model = fct_inorder(model)) |> 
  ggplot(aes(x = range, y = logratio)) + 
  geom_point() + 
  geom_line(aes(y = pred), color = "red") + 
  facet_wrap(~model)
```

![](cross_validation_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
rmse(linear_mod, test_df)
```

    ## [1] 0.136171

``` r
rmse(smooth_mod, test_df)
```

    ## [1] 0.09012073

``` r
rmse(wiggly_mod, test_df)
```

    ## [1] 0.09891565

``` r
cv_df = 
  crossv_mc(lidar_df, 100) |>
  mutate(
    train = map(train, as_tibble),
    test = map(test, as_tibble)
  )
```

did this work? Yes!

``` r
cv_df |> pull(train) |> nth(3)
```

    ## # A tibble: 176 × 3
    ##    range logratio    id
    ##    <dbl>    <dbl> <int>
    ##  1   390  -0.0504     1
    ##  2   391  -0.0601     2
    ##  3   393  -0.0419     3
    ##  4   394  -0.0510     4
    ##  5   396  -0.0599     5
    ##  6   397  -0.0284     6
    ##  7   402  -0.0294     9
    ##  8   403  -0.0395    10
    ##  9   408  -0.0312    13
    ## 10   409  -0.0382    14
    ## # ℹ 166 more rows

let’s fit the model over and over

``` r
lidar_lm = function(df){
  lm(logratio ~ range, data = df)
}
```

``` r
cv_df |>
  mutate(
    linear_fits = map(train, \(df) lm(logratio ~ range, data = df))
  ) |>
  mutate(
    rmse_linear = map2(linear_fits, test, rmse)
  )
```

    ## # A tibble: 100 × 5
    ##    train              test              .id   linear_fits rmse_linear
    ##    <list>             <list>            <chr> <list>      <list>     
    ##  1 <tibble [176 × 3]> <tibble [45 × 3]> 001   <lm>        <dbl [1]>  
    ##  2 <tibble [176 × 3]> <tibble [45 × 3]> 002   <lm>        <dbl [1]>  
    ##  3 <tibble [176 × 3]> <tibble [45 × 3]> 003   <lm>        <dbl [1]>  
    ##  4 <tibble [176 × 3]> <tibble [45 × 3]> 004   <lm>        <dbl [1]>  
    ##  5 <tibble [176 × 3]> <tibble [45 × 3]> 005   <lm>        <dbl [1]>  
    ##  6 <tibble [176 × 3]> <tibble [45 × 3]> 006   <lm>        <dbl [1]>  
    ##  7 <tibble [176 × 3]> <tibble [45 × 3]> 007   <lm>        <dbl [1]>  
    ##  8 <tibble [176 × 3]> <tibble [45 × 3]> 008   <lm>        <dbl [1]>  
    ##  9 <tibble [176 × 3]> <tibble [45 × 3]> 009   <lm>        <dbl [1]>  
    ## 10 <tibble [176 × 3]> <tibble [45 × 3]> 010   <lm>        <dbl [1]>  
    ## # ℹ 90 more rows

``` r
cv_df |>
  mutate(
    linear_fits = map(train, \(df) lm(logratio ~ range, data = df))
  ) |>
  mutate(
    rmse_linear = map2_dbl(linear_fits, test, rmse)
  )
```

    ## # A tibble: 100 × 5
    ##    train              test              .id   linear_fits rmse_linear
    ##    <list>             <list>            <chr> <list>            <dbl>
    ##  1 <tibble [176 × 3]> <tibble [45 × 3]> 001   <lm>              0.152
    ##  2 <tibble [176 × 3]> <tibble [45 × 3]> 002   <lm>              0.130
    ##  3 <tibble [176 × 3]> <tibble [45 × 3]> 003   <lm>              0.124
    ##  4 <tibble [176 × 3]> <tibble [45 × 3]> 004   <lm>              0.131
    ##  5 <tibble [176 × 3]> <tibble [45 × 3]> 005   <lm>              0.151
    ##  6 <tibble [176 × 3]> <tibble [45 × 3]> 006   <lm>              0.110
    ##  7 <tibble [176 × 3]> <tibble [45 × 3]> 007   <lm>              0.134
    ##  8 <tibble [176 × 3]> <tibble [45 × 3]> 008   <lm>              0.140
    ##  9 <tibble [176 × 3]> <tibble [45 × 3]> 009   <lm>              0.138
    ## 10 <tibble [176 × 3]> <tibble [45 × 3]> 010   <lm>              0.136
    ## # ℹ 90 more rows
